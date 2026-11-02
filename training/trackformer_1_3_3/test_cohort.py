import unittest,hashlib
from prepare_colab import catalog,select_cached,times


class CohortTests(unittest.TestCase):
    def test_expansion_uses_only_cached_frames_and_no_development_windows(self):
        d=catalog();ranked=sorted((r for r in d['records'] if r['split']=='train'),
            key=lambda r:hashlib.sha256(('131:'+r['id']).encode()).hexdigest())
        first=[];seen=set()
        for r in ranked:
            if r['storm_id'] not in seen:first.append(r);seen.add(r['storm_id'])
        dev=[min((r for r in d['records'] if r['split']=='development' and r['storm_id']==sid),key=lambda r:r['issue_seconds'])
            for sid in sorted({r['storm_id'] for r in d['records'] if r['split']=='development'})]
        old=dict(train=first[:64],development=dev);new=select_cached(d,old)
        self.assertEqual(len(new['train']),186);self.assertEqual(len({r['storm_id'] for r in new['train']}),84)
        self.assertEqual(new['development'],old['development'])
        available=set().union(*(times(r) for rows in old.values() for r in rows))
        dev_times=set().union(*(times(r) for r in dev))
        for r in new['train']:
            self.assertTrue(times(r)<=available);self.assertFalse(times(r)&dev_times)


if __name__=='__main__':unittest.main()
