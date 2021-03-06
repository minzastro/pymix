# encoding: utf-8
#
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Author: Kyle Lahnakoski (kyle@lahnakoski.com)
#

from __future__ import unicode_literals
from __future__ import division

from collections import Iterable
from types import GeneratorType
from ..struct import nvl
from ..env.logs import Log
from ..thread.threads import Queue, Thread

DEBUG = False


class Multithread(object):
    """
    SIMPLE SEMANTICS FOR SYMMETRIC MULTITHREADING
    PASS A SET OF functions TO BE EXECUTED (ONE PER THREAD)
    SET outbound==False TO SIMPLY THROW AWAY RETURN VALUES, IF ANY
    threads - IF functions IS NOT AN ARRAY, THEN threads  IS USED TO MAKE AN ARRAY
    THE inbound QUEUE IS EXPECTING dicts, EACH dict IS USED AS kwargs TO GIVEN functions
    """

    def __init__(self, functions, threads=None, outbound=None, silent_queues=None):
        if outbound is None:
            self.outbound = Queue(silent=silent_queues)
        elif outbound is False:
            self.outbound = None
        else:
            self.outbound = outbound

        self.inbound = Queue(silent=silent_queues)

        # MAKE THREADS
        if isinstance(functions, Iterable):
            if threads:
                Log.error("do not know how to handle an array of functions AND a thread multiplier")
            self.threads = []
            for t, f in enumerate(functions):
                thread = worker_thread("worker " + unicode(t), self.inbound, self.outbound, f)
                self.threads.append(thread)
        else:
            # ASSUME functions IS A SINGLE FUNCTION
            self.threads = []
            for t in range(nvl(threads, 1)):
                thread = worker_thread("worker " + unicode(t), self.inbound, self.outbound, functions)
                self.threads.append(thread)

    def __enter__(self):
        return self

    # WAIT FOR ALL QUEUED WORK TO BE DONE BEFORE RETURNING
    def __exit__(self, type, value, traceback):
        try:
            if isinstance(value, Exception):
                self.inbound.close()
            self.inbound.add(Thread.STOP)
            self.join()
        except Exception, e:
            Log.warning("Problem sending stops", e)


    # IF YOU SENT A stop(), OR Thread.STOP, YOU MAY WAIT FOR SHUTDOWN
    def join(self):
        try:
            # WAIT FOR FINISH
            for t in self.threads:
                t.join()
        except (KeyboardInterrupt, SystemExit):
            Log.note("Shutdow Started, please be patient")
        except Exception, e:
            Log.error("Unusual shutdown!", e)
        finally:
            for t in self.threads:
                t.keep_running = False
            self.inbound.close()
            if self.outbound:
                self.outbound.close()
            for t in self.threads:
                t.join()

    def execute(self, requests):
        """
        RETURN A GENERATOR THAT HAS len(requests) RESULTS (ANY ORDER)
        EXPECTING requests TO BE A list OF dicts, EACH dict IS USED AS kwargs TO GIVEN functions
        """
        if not isinstance(requests,(list, tuple, GeneratorType)):
            Log.error("Expecting requests to be a list or generator", offset=1)

        # FILL QUEUE WITH WORK
        self.inbound.extend(requests)

        num = len(requests)

        def output():
            for i in xrange(num):
                result = self.outbound.pop()
                if "exception" in result:
                    raise result["exception"]
                else:
                    yield result["response"]

        if self.outbound is not None:
            return output()
        else:
            return

    # EXTERNAL COMMAND THAT RETURNS IMMEDIATELY
    def stop(self):
        self.inbound.close() # SEND STOPS TO WAKE UP THE WORKERS WAITING ON inbound.pop()
        for t in self.threads:
            t.keep_running = False


class worker_thread(Thread):
    # in_queue MUST CONTAIN HASH OF PARAMETERS FOR load()
    def __init__(self, name, in_queue, out_queue, function):
        Thread.__init__(self, name, self.event_loop)
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.function = function
        self.num_runs = 0
        self.start()

    def event_loop(self, please_stop):
        got_stop = False
        while not please_stop.is_go():
            request = self.in_queue.pop()
            if request == Thread.STOP:
                got_stop = True
                if self.in_queue.queue:
                    Log.warning("programmer error, queue not empty. {{num}} requests lost:\n{{requests}}", {
                        "num": len(self.in_queue.queue),
                        "requests": self.in_queue.queue[:5:] + self.in_queue.queue[-5::]
                    })
                break
            if please_stop.is_go():
                break

            try:
                if DEBUG and hasattr(self.function, "func_name"):
                    Log.note("run {{function}}", {"function": self.function.func_name})
                result = self.function(**request)
                if self.out_queue != None:
                    self.out_queue.add({"response": result})
            except Exception, e:
                Log.warning("Can not execute with params={{params}}", {"params": request}, e)
                if self.out_queue != None:
                    self.out_queue.add({"exception": e})
            finally:
                self.num_runs += 1

        please_stop.go()
        del self.function

        if self.num_runs == 0:
            Log.warning("{{name}} thread did no work", {"name": self.name})
        if DEBUG and self.num_runs != 1:
            Log.note("{{name}} thread did {{num}} units of work", {
                "name": self.name,
                "num": self.num_runs
            })
        if got_stop and self.in_queue.queue:
            Log.warning("multithread programmer error, queue not empty. {{num}} requests lost", {"num": len(self.in_queue.queue)})
        if DEBUG:
            Log.note("{{thread}} DONE", {"thread": self.name})

