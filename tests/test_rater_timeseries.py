from __future__ import annotations
import math
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from contextlib import closing

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'tools'))
from plot_rater_timeseries import load_series, line_points, write_outputs, smooth_values


class RaterTimeseriesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db = Path(self.temp.name) / 'ratings.sqlite'
        with closing(sqlite3.connect(self.db)) as c:
            c.executescript('''
                CREATE TABLE videos(id INTEGER, path TEXT);
                CREATE TABLE windows(id INTEGER,video_id INTEGER,t_start REAL,t_end REAL);
                CREATE TABLE segments(id INTEGER,window_id INTEGER,track_id INTEGER);
                CREATE TABLE labels(segment_id INTEGER,rater TEXT,score INTEGER);
                CREATE TABLE window_skips(window_id INTEGER,rater TEXT);
                CREATE TABLE excluded_segments(segment_id INTEGER);
                INSERT INTO videos VALUES(1,'lesson.mp4'),(2,'second.mp4');
                INSERT INTO windows VALUES(1,1,0,5),(2,1,5,10),(3,1,15,20),(4,2,0,5);
                INSERT INTO segments VALUES(1,1,10),(2,1,11),(3,1,12),(4,2,10),(5,3,10),(6,4,10);
                INSERT INTO labels VALUES(1,'田中',3),(2,'田中',5),(3,'田中',7),
                    (1,'鈴木',6),(4,'田中',4),(4,'鈴木',7),(5,'田中',5),(6,'田中',2);
                INSERT INTO excluded_segments VALUES(3);
                INSERT INTO window_skips VALUES(2,'田中');
            ''')

    def test_mean_skip_exclusion_missing_and_readonly(self):
        before = self.db.read_bytes()
        rows, videos = load_series(self.db)
        self.assertEqual(videos[2], 'second.mp4')
        by = {(r['window_id'],r['rater']):r for r in rows}
        self.assertEqual(by[1,'田中']['score'],4)
        self.assertEqual(by[1,'田中']['n_labeled'],2)
        self.assertIsNone(by[2,'田中']['score'])
        self.assertEqual(by[2,'鈴木']['score'],7)
        self.assertIsNone(by[3,'鈴木']['score'])
        self.assertEqual(before,self.db.read_bytes())

    def test_single_rater_and_track(self):
        rows,_=load_series(self.db,['田中'],10)
        self.assertEqual(rows[0]['score'],3)
        self.assertEqual({r['rater'] for r in rows},{'田中'})
        with self.assertRaises(ValueError): load_series(self.db,['不明'])
        with self.assertRaises(ValueError): load_series(self.db,track_id=999)

    def test_gaps_and_unrated_are_not_connected(self):
        rows,_=load_series(self.db,['田中'])
        xs,ys=line_points([r for r in rows if r['video_id']==1])
        self.assertEqual(len(xs),4)
        self.assertAlmostEqual(xs[0],2.5/60)
        self.assertTrue(math.isnan(ys[1]))
        self.assertTrue(math.isnan(ys[2]))

    def test_centered_mean_and_boundary_windows(self):
        self.assertEqual(smooth_values([1, 7, 1, 7, 1], 3), [4, 3, 5, 3, 4])
        self.assertEqual(smooth_values([2], 5), [2])
        self.assertEqual(smooth_values([1, 7, 1], 1), [1, 7, 1])
        for invalid in (0, -1, 2):
            with self.assertRaises(ValueError): smooth_values([], invalid)

    def test_smoothing_preserves_missing_and_gap_boundaries(self):
        values = [1, 3, math.nan, 7, 7]
        result = smooth_values(values, 5)
        self.assertEqual(result[:2], [2, 2])
        self.assertTrue(math.isnan(result[2]))
        self.assertEqual(result[3:], [7, 7])
        rows = [dict(t_start=0, t_end=5, score=1),
                dict(t_start=10, t_end=15, score=7)]
        _, ys = line_points(rows)
        smoothed = smooth_values(ys, 3)
        self.assertEqual(smoothed[0], 1)
        self.assertTrue(math.isnan(smoothed[1]))
        self.assertEqual(smoothed[2], 7)

    def test_smoothing_leaves_csv_and_input_unchanged(self):
        rows, videos = load_series(self.db)
        original = [dict(row) for row in rows]
        raw_dir = Path(self.temp.name) / 'raw'
        smooth_dir = Path(self.temp.name) / 'smooth'
        write_outputs(rows, videos, raw_dir, 'raw')
        write_outputs(rows, videos, smooth_dir, 'smooth', smooth_window=3)
        self.assertEqual(rows, original)
        self.assertEqual((raw_dir/'ratings_timeseries.csv').read_bytes(),
                         (smooth_dir/'ratings_timeseries.csv').read_bytes())

    def test_output_multiple_videos_and_single_rater(self):
        for names,folder in [(None,'multiple'),(['田中'],'single')]:
            rows,videos=load_series(self.db,names)
            outputs = write_outputs(rows, videos, Path(self.temp.name) / folder,
                                    '評価者別集中度', db_stem=self.db.stem)
            self.assertEqual([p.name for p in outputs], [
                'ratings_ratings_timeseries.csv',
                'ratings_concentration_video_1.png', 'ratings_concentration_video_1.svg',
                'ratings_concentration_video_2.png', 'ratings_concentration_video_2.svg',
            ])
            self.assertTrue(all(p.is_file() and p.stat().st_size>100 for p in outputs))


if __name__=='__main__':
    unittest.main()
