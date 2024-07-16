#!/usr/bin/env python3
'''
Tworoutine Tests
~~~~~~~~~~~~~~~~

"Tworoutines" are the Python-3 equivalent of pydfmux's
"asynchronous/parallelizable" infrastructure. The following tets ensure that the
tworoutines API behaves within the limits imposed by Python 3 event loops.
'''

import unittest
import asyncio
import hidfmux


@hidfmux.tworoutine
async def async_return(x):
    """Trivial tworoutine definition that just returns its argument."""
    return x


class TworoutineFunctionTestCase(unittest.TestCase):
    def test_function_sync_call(self):
        """Synchronous coroutine function dispatch"""

        self.assertEqual(async_return(100), 100)

    def test_function_async_call(self):
        """Asynchronous coroutine function dispatch"""

        co = (~async_return)(200)
        self.assertTrue(asyncio.iscoroutine(co))
        self.assertEqual(asyncio.get_event_loop().run_until_complete(co), 200)


class TworoutineMethodTestCase(unittest.TestCase):
    @hidfmux.tworoutine
    async def async_return(self, x):
        return x

    def test_sync(self):
        """Synchronous coroutine method dispatch"""

        self.assertEqual(self.async_return(300), 300)

    def test_async(self):
        """Asynchronous coroutine method dispatch"""

        co = (~self.async_return)(300)
        self.assertTrue(asyncio.iscoroutine(co))
        self.assertEqual(asyncio.get_event_loop().run_until_complete(co), 300)


class TworoutineClassMethodTestCase(unittest.TestCase):
    @classmethod
    @hidfmux.tworoutine
    async def async_return(self, x):
        return x

    def test_sync(self):
        """Synchronous coroutine classmethod dispatch"""

        # Coroutines may be invoked on both the class and the instance
        # (https://docs.python.org/3/library/functions.html#classmethod)
        self.assertEqual(self.async_return(300), 300)
        self.assertEqual(self.__class__.async_return(300), 300)

    # FIXME: @classmethod obscures the ~ operator we need
    # def test_async(self):
    #    """Asynchronous coroutine classmethod dispatch"""
    #
    #    co = (~ClassWithTworoutines.async_return)(300)
    #    self.assertTrue(asyncio.iscoroutine(co))
    #    self.assertEqual(asyncio.get_event_loop().run_until_complete(co), 300)


class TworoutineStaticMethodTestCase(unittest.TestCase):
    @staticmethod
    @hidfmux.tworoutine
    async def async_return(x):
        return x

    def test_sync(self):
        """Synchronous coroutine staticmethod dispatch"""

        self.assertEqual(self.async_return(300), 300)

    def test_async(self):
        """Asynchronous coroutine staticmethod dispatch"""

        co = (~self.async_return)(300)
        self.assertTrue(asyncio.iscoroutine(co))
        self.assertEqual(asyncio.get_event_loop().run_until_complete(co), 300)


if __name__ == "__main__":
    unittest.main()
