"""Mac data preparation with three concurrent source-variable reads.

Uses the same plan, fields, validation and resume records as the serial preparer.
This utility is not called by the Drive-only Colab notebook.
"""
from concurrent.futures import ThreadPoolExecutor

import v164_prepare_data as prep


class ConcurrentEra5Source(prep.Era5Source):
    _pending = None

    def read(self, name, ns, levels=None):
        key = (name, int(ns), tuple(levels) if levels is not None else None)
        if self._pending is not None and key in self._pending:
            return self._pending.pop(key).result()
        return super().read(name, ns, levels)

    def frame(self, ns):
        jobs = [(name, None) for name in prep.SURFACE]
        jobs += [(name, tuple(prep.trainer.LEVELS)) for name in prep.UPPER]
        with ThreadPoolExecutor(max_workers=3) as pool:
            self._pending = {(name, int(ns), levels): pool.submit(super().read, name, ns, levels)
                             for name, levels in jobs}
            try:
                return super().frame(ns)
            finally:
                for future in self._pending.values():
                    future.cancel()
                self._pending = None


if __name__ == '__main__':
    prep.Era5Source = ConcurrentEra5Source
    prep.main()
