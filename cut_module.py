# Apply the selection cuts of choice, no optimization is done in this fill

import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import math
#from mom_components import mom_components

class CutClass:

    def __init__(self,  opt = 'SU2020', use_CRV = True):
        self.use_CRV = use_CRV
        self.Event_cut = {} # Cut applied at the event level
        self.Track_cut = {} # Cut applied at the track level
        self.Trksegs_cut = {}# Cut applied at the trksegs level
        self.CRV_cut = {}   # Cut applied based on CRV coincidence
        self.MC_cut = {}    # Cut applied on the MC data
        if opt == 'SU2020':
            self.Event_cut = {
            }
            self.Track_cut = {
                "'trk','trk.pdg'": 11,   # trk uses e- hypothesis for Kalman fit
                "'trk','trk.nactive'": [20, float('inf')], # active hits in the tracker
                "'trkqual','trkqual.result'": [0.2, float('inf')]
            }
            self.Trksegs_cut = {
                "'trksegs','sid'": 0,   # Look at the track at the entrance of the tracker
                "'trksegs','mom','fCoordinates','fZ'": [0, float('inf')],   # Look at downstream tracks
                "'trksegs','time'": [640., 1650], #inTimeWindow
                "'trksegpars_lh','t0err'": [0, 0.9], #intimeErr
                "'trksegpars_lh','maxr'": [450., 680.], #inMaxRCut
            }
            self.CRV_cut = {
                "crv_coincidence" : 150. # cut anything with a track - crv time difference less than this
            }

    def ApplyCut(self, array_trk, array_crv, verbose =0):
        """ function applies cuts to MDS1 """
        if (verbose == 1):
          print("\nApplying cuts\n")
          print("# of events before cut: ", ak.num(array_trk, axis=0))
          print("# of tracks before cut: ", ak.count(array_trk['trk','trk.status']))

        array_cut = ak.copy(array_trk)  # Use copy to keep the initial array untouched

        # Event level cut: TODO when there is one
        
        # Track level cut
        for key, value in self.Track_cut.items():
            if (verbose == 1):
              print(eval(key))

            if type(value) == int:
                mask = array_trk[eval(key)] == value
            elif len(value) == 2:
                mask = (array_trk[eval(key)] >= value[0]) & (array_trk[eval(key)] <= value[1])

            ApplyMaskTrk(array_cut, mask)
            #array_trk[eval(key)].show()
            #array_cut[eval(key)].show()
            if (verbose == 1):
              print("# of tracks passing this cut: ", ak.count(array_cut['trk','trk.status']))
            
        # Track segments level cut
        for key, value in self.Trksegs_cut.items():
            if (verbose == 1):
              print(eval(key))

            if type(value) == int:
                mask = array_trk[eval(key)] == value
            elif len(value) == 2:
                mask = (array_trk[eval(key)] >= value[0]) & (array_trk[eval(key)] <= value[1])

            ApplyMaskTrkVec(array_cut, mask)
            #mask.show()
            #array_trk[eval(key)].show()
            #array_cut[eval(key)].show()

            if (verbose == 1):
              print("# of trksegs passing this cut: ", ak.count(array_cut['trksegs','time']))

        # CRV cuts
        if self.use_CRV:
            # TODO to decide
            # - Look only at first trkseg (current implementation), all trksegs, trkseg for specific sid?
            # - Look at trkseg before or after applying selection? (This doesn't matter for current implementation, might matter if we changed trkseg choice)

            # Get array where for each event, for each track, we have all combinations of leading trkseg time and crv time
            combinations = ak.cartesian([ak.firsts(array_trk['trksegs','time'],axis=-1),array_crv['crvcoincs','crvcoincs.time']],axis=1,nested=True)
            trksegtime, crvtime = ak.unzip(combinations)
            mintimediff = ak.min(abs(trksegtime-crvtime),axis=-1)
            # True if mintimediff >= cut or no crv hit; False if mintimediff < cut
            mask = ak.fill_none((mintimediff >= self.CRV_cut.get('crv_coincidence')),True)
            ApplyMaskTrk(array_cut, mask)
        if (verbose == 1):
          print("# of trksegs after all the cuts: ", ak.count(array_cut['trksegs','time']))
        
        return array_cut
    
    def ApplyCutMCSim(self, array_cut, verbose = 0):
        first_sim = array_cut["trkmcsim"][..., 0]
        array_cut = ak.copy(array_cut)
        array_cut['trkmcsim'] = first_sim
        return array_cut


    def ApplyCutMC(self, array_mc, array_trk_cut, verbose = 0):
        """ Apply the trk cut on the MC array"""
        """ Reproduce the combination of all masks applied on the trk array and apply it on the MC array """
        if (verbose == 1):
          print("\nApplying cuts on MC array\n")
          print("# of events before cut: ", ak.num(array_mc, axis=0))
          print("# of tracks before cut: ", ak.count(array_mc['trkmc','trkmc.valid']))
        array_mc_cut = ak.copy(array_mc)

        # Event level cut: TODO when there is one

        # Track level cut
        mask = ~(ak.is_none(array_trk_cut['trk','trk.nactive'], axis=1))
        ApplyMaskTrk(array_mc_cut, mask)
        mask.show()
        array_mc['trkmc','trkmc.nactive'].show()
        array_mc_cut['trkmc','trkmc.nactive'].show()

        # Track segments level cut
        mask = ~(ak.is_none(array_trk_cut['trksegs','time'], axis=2))
        ApplyMaskTrkVec(array_mc_cut, mask)
        mask.show()
        array_mc['trksegsmc','time'].show()
        array_mc_cut['trksegsmc','time'].show()

        #print("# of events after all the cuts: ", ak.num(array_trk, axis=0)
        if (verbose == 1):
          print("# of tracks after all the cuts: ", ak.count(array_mc_cut['trksegsmc','time']))

    def CategorizeTracks(self, array_mc, mismatch=False, verbose = 0):
        if (verbose == 1):
          print('Doing track categorization')

        array_tmp = ak.copy(array_mc)

        mask = (array_tmp['trkmcsim']['rank'] == 0) & (array_tmp['trkmcsim']['nhits'] > 0)
        ApplyMaskTrkVec(array_tmp, mask)

        if mismatch:
            pStartCode = ak.max(ak.flatten(array_tmp['trkmcsim']['startCode'],axis=2),axis=1,mask_identity=True)
            pGenCode = ak.max(ak.flatten(array_tmp['trkmcsim']['gen'],axis=2),axis=1,mask_identity=True)
        else:
            pStartCode = ak.flatten(ak.drop_none(array_tmp['trkmcsim']['startCode']),axis=2,mask_identity=True)
            pGenCode = ak.flatten(ak.drop_none(array_tmp['trkmcsim']['gen']),axis=2,mask_identity=True)
        pStartCode = ak.fill_none(pStartCode,-1)
        pGenCode = ak.fill_none(pGenCode,-1)
        
        categories = ak.zeros_like(pStartCode)
        for icat, idict in enumerate(mom_components.values()):
            startCodes = idict['startCode']
            genCodes = idict['genCode']
            goodCode = ak.zeros_like(pStartCode,dtype=bool)
            for startCode in startCodes:
                for genCode in genCodes:
                    goodStartCode = ak.ones_like(pStartCode,dtype=bool) if startCode is None else (pStartCode == startCode)
                    goodGenCode = ak.ones_like(pGenCode,dtype=bool) if genCode is None else (pGenCode == genCode)
                    goodCode = goodCode | (goodStartCode & goodGenCode)
            categories = categories + (icat+1) * (goodCode)

        return categories
    
    
def ApplyMaskTrk(array_cut, mask):
    """ Apply the mask onto all branches of the array, broadcasting to vector-type branches """
    for branch in ak.fields(array_cut):
        for leaf in ak.fields(array_cut[branch]):
            if array_cut[branch].layout.minmax_depth[1] > 2:
                mask_vec = ak.broadcast_arrays(array_cut[branch],mask,depth_limit=3)[1]
                array_cut[branch,leaf] = array_cut[branch,leaf].mask[mask_vec]
            else:
                array_cut[branch,leaf] = array_cut[branch,leaf].mask[mask]

    return array_cut


def ApplyMaskTrkVec(array_cut, mask):
    """ Apply the mask onto each vector-type track branch of the array with the same structure (trkseg -> trkseg, trkcmsim -> trkmcsim) """
    for branch in ak.fields(array_cut):
        if array_cut[branch].layout.minmax_depth[1] > 2:
            for leaf in ak.fields(array_cut[branch]):
                if ak.almost_equal(ak.num(mask,axis=2),ak.num(array_cut[branch][leaf],axis=2)):
                    array_cut[branch,leaf] = array_cut[branch,leaf].mask[mask]

    return array_cut

