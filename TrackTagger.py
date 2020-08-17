import numpy as np
from scipy.optimize import linear_sum_assignment
import cv2

EMB_DIM = 4

def det2pnt(dets):
    dets = dets.reshape(-1, 4)
    cx = 0.5*(dets[:, 0]+dets[:, 2]).reshape(-1, 1)
    cy = 0.5*(dets[:, 1]+dets[:, 3]).reshape(-1, 1)   
    x = np.hstack((cx, cy)) 
    return x

class MyTracker(object):
    count = 0
    def __init__(self, x=None):
        self.next_track_id = 0
        self.lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        self.feature_params = dict( maxCorners = 500,
                            qualityLevel = 0.3,
                            minDistance = 7,
                            blockSize = 7)   
        self.track_len = 10
        self.detect_interval = 5        
        self.frame_prev = None        
        self.x = x
        self.radius = 5
        self.id = MyTracker.count
        MyTracker.count += 1
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.time_since_update = 0
        
    def getPixCands(self):
        x_min = self.x[0] - self.radius
        x_max = self.x[0] + self.radius + 1
        y_min = self.x[1] - self.radius
        y_max = self.x[1] + self.radius + 1
        x = np.arange(x_min, x_max, 1)
        y = np.arange(y_min, y_max, 1)
        X, Y = np.meshgrid(x, y)
        X = X.reshape(-1, 1)
        Y = Y.reshape(-1, 1)
        w = np.hstack((X, Y)).reshape(-1, 1, 2).astype(np.float32)
        return w
    
    def step(self, frame, det=None):
        if det is None:
            w = self.getPixCands()
            p1, _st, _err = cv2.calcOpticalFlowPyrLK(self.frame_prev, frame, w, None, **self.lk_params)
            p0r, _st, _err = cv2.calcOpticalFlowPyrLK(frame, self.frame_prev, p1, None, **self.lk_params)
            d = np.linalg.norm(p1-p0r)
            ii = np.argmin(d)
            self.x = w[ii, 0, :].reshape(-1, 2)

            self.age += 1
            if (self.time_since_update > 0):
                self.hit_streak = 0
            self.time_since_update += 1
        else:
            self.x = det2pnt(det)
            self.frame_prev = frame        
            self.time_since_update = 0
            self.hits += 1
            self.hit_streak += 1


class TrackTagger(object):
    def __init__(self, max_age, min_hits):
        self.tracker = MyTracker()        
        self._count = 0        
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks = []
        self.next_id = 1                

    def assigment(self, dets):

        if(len(self.tracks)==0):
            return np.empty((0,2),dtype=int), np.arange(len(dets)), np.empty((0,5),dtype=int)                    
        trks = np.zeros( (len(self.tracks), 2), dtype=np.float32)
        for i, t in enumerate(self.tracks):
            trks[i, :] = t.x
        x = det2pnt(dets)
        score = trks.dot(x.T)
        matched = linear_sum_assignment(-score)
        
        unmatched_detections = []
        for d, det in enumerate(dets):
            if(d not in matched[0]):
                unmatched_detections.append(d)
        unmatched_trackers = []
        for t,trk in enumerate(self.tracks):
            if(t not in matched[1]):
                unmatched_trackers.append(t)

        return matched, unmatched_detections, unmatched_trackers
    
    def step(self, frame, dets):
        self._count += 1
        matched, unmatched_detections, unmatched_tracks = self.assigment(dets)
        #update matched trackers with assigned detections
        if len(matched) > 0:
            matched_t = matched[0]
            matched_d = matched[1]
            for d, t in zip(matched_t, matched_d):
                self.tracks[t].step(frame, dets[d, :])
                
        for i in unmatched_tracks:
            self.tracks[i].step(frame, None)
                    
        #create and initialise new trackers for unmatched detections
        x = det2pnt(dets)
        for i in unmatched_detections:            
            trk = MyTracker(x[i, :])            
            self.tracks.append(trk)

        res_trks = []        
        i = len(self.tracks)
        for trk in reversed(self.tracks):
            d = trk.x.ravel()
            if (trk.time_since_update < 1):
                res_trks.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            if(trk.time_since_update > self.max_age):
                self.tracks.pop(i)

        return res_trks
        


if __name__ == "__main__":
    dets = np.random.randn(5, EMB_DIM)
    traker = TrackTagger(max_age=10, min_hits=1)
    frame = None
    trks = traker.step(frame, dets)
    trks = traker.step(frame, dets)    
        