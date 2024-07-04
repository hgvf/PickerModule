import torch
import numpy as np
import time

class EEW_sta(object):
    def __init__(self, station, channel, network, location, gain):
        self.station = station
        self.channel = channel
        self.network = network
        self.location = location
        self.gain = gain
        self.samp = 100.0
        self.dt = 1 / self.samp
        
        self._init_params()
                
    def receive_new_waveform(self, waveform, nsamp):        
        self.waveform = waveform

        self._update_avd(nsamp)
    
    def _update_avd(self, nsamp):
        for samp in range(nsamp):
            self.avd0 = self.a
            self.avd1 = self.v
            self.avd2 = self.d
            
            self.v = self.waveform[samp]    # fetch trace data point by point
            self.v = self.v / self.gain       #  for gain factor   
            self.ave = self.ave * 999.0 / 1000.0 + self.v / 1000.0 # remove instrument response
            self.v = self.v - (self.ave) # remove offset
    
            # --------------For Recursive Filter  high pass 2 poles at 0.075 Hz #
            OUTPUT = self.B0* self.v + self.B1 * self.X1 + self.B2 * self.X2
            OUTPUT = OUTPUT - ( self.A1 * self.Y1 + self.A2 * self.Y2)
            self.Y2 = self.Y1
            self.Y1 = OUTPUT
            self.X2 = self.X1
            self.X1 = self.v
            self.v = OUTPUT
            # --------------End of Recursive Filter #
    
            # from ACC to VEL #
            acc = self.v
            self.v = (acc + self.acc0) * self.dt / 2.0 + self.vel0
            self.acc0 = acc
            self.vel0 = self.v
            self.ave0 = self.ave0 * (10. - self.dt) / 10.0 + self.v * self.dt / 10.0
            self.v = self.v - self.ave0

            # --------------For Recursive Filter  high pass 2 poles at 0.075 Hz #
            OUTPUT = self.B0 * self.v + self.B1 * self.XX1 + self.B2 * self.XX2
            OUTPUT = OUTPUT - ( self.A1 * self.YY1 + self.A2 * self.YY2 )
            self.YY2 = self.YY1
            self.YY1 = OUTPUT
            self.XX2 = self.XX1
            self.XX1 = self.v
            self.v = OUTPUT
            # --------------End of Recursive Filter #
        
            # --------------------------------------------------------------------------------------------
            self.a = (self.v - self.avd1) / self.dt

            acc_vel = self.v
            self.d = (acc_vel + self.acc0_vel) * self.dt / 2.0 + self.vel0_dis
            self.acc0_vel = acc_vel
            self.vel0_dis = self.d
            self.ave1 = self.ave1 * (10. - self.dt) / 10.0 + self.d * self.dt / 10.0
            self.d = self.d - self.ave1

            # --------------For Recursive Filter  high pass 2 poles at 0.075 Hz #
            OUTPUT = self.B0 * self.d + self.B1 * self.x1 + self.B2 * self.x2
            OUTPUT = OUTPUT - ( self.A1 * self.y1 + self.A2 * self.y2 )
            self.y2 = self.y1
            self.y1 = OUTPUT
            self.x2 = self.x1
            self.x1 = self.d
            self.d = OUTPUT
            # --------------End of Recursive Filter #
            self.pa = max(self.pa, abs(self.a))
            self.pv = max(self.pv, abs(self.v))
            self.pd = max(self.pd, abs(self.d))
            
    def get_Pavd(self):
        pa, pv, pd = round(self.pa, 6), round(self.pv, 6), round(self.pd, 6)

        return pa, pv, pd
    
    def reset_pavd(self):
        self.pa = 0
        self.pv = 0
        self.pd = 0
        
    def _init_params(self):
        self.X1 = 0
        self.X2 = 0
        self.Y1 = 0
        self.Y2 = 0
        self.XX1 = 0
        self.XX2 = 0
        self.YY1 = 0
        self.YY2 = 0
        self.x1 = 0
        self.x2 = 0
        self.y1 = 0
        self.y2 = 0
        self.ave  = 0
        self.ave0 = 0
        self.ave1 = 0
        self.acc0 = 0
        self.vel0 = 0
        self.acc0_vel = 0
        self.vel0_dis = 0
        self.vvsum = 0
        self.ddsum = 0
        self.a = 0
        self.v = 0
        self.d = 0
        self.ddv = 0
        self.pa = -1000
        self.pv = -1000
        self.pd = -1000
        self.tc = 0
        self.avd0 = 0.0
        self.avd1 = 0.0
        self.avd2 = 0.0
        
        self.B0 = 0.9966734
        self.B1 = -1.993347
        self.B2 = 0.9966734
        self.A1 = -1.993336
        self.A2 = 0.9933579

def create_station(sta, gain):
    # parameters initialization
    sta['X1'] = 0
    sta['X2'] = 0
    sta['Y1'] = 0
    sta['Y2'] = 0
    sta['XX1'] = 0
    sta['XX2'] = 0
    sta['YY1'] = 0
    sta['YY2'] = 0
    sta['x1'] = 0
    sta['x2'] = 0
    sta['y1'] = 0
    sta['y2'] = 0
    sta['ave'] = 0
    sta['ave0'] = 0
    sta['ave1'] = 0
    sta['acc0'] = 0
    sta['vel0'] = 0
    sta['acc0_vel'] = 0
    sta['vel0_dis'] = 0
    sta['vvsum'] = 0
    sta['ddsum'] = 0
    sta['a'] = 0
    sta['v'] = 0
    sta['d'] = 0
    sta['ddv'] = 0
    sta['pa'] = -1000
    sta['pv'] = -1000
    sta['pd'] = -1000
    sta['tc'] = 0
    sta['avd0'] = 0.0
    sta['avd1'] = 0.0
    sta['avd2'] = 0.0
        
    sta['B0'] = 0.9966734
    sta['B1'] = -1.993347
    sta['B2'] = 0.9966734
    sta['A1'] = -1.993336
    sta['A2'] = 0.9933579
    sta['gain'] = gain
    sta['dt'] = 1 / 100.0

    return sta

def update_avd(sta, waveform, nsamp):
    for samp in range(nsamp):
        sta['avd0'] = sta['a']
        sta['avd1'] = sta['v']
        sta['avd2'] = sta['d']
        
        sta['v'] = waveform[samp]    # fetch trace data point by point
        sta['v'] = sta['v'] / sta['gain']       #  for gain factor   
        sta['ave'] = sta['ave'] * 999.0 / 1000.0 + sta['v'] / 1000.0 # remove instrument response
        sta['v'] = sta['v'] - (sta['ave']) # remove offset

        # --------------For Recursive Filter  high pass 2 poles at 0.075 Hz #
        OUTPUT = sta['B0']* sta['v'] + sta['B1'] * sta['X1'] + sta['B2'] * sta['X2']
        OUTPUT = OUTPUT - ( sta['A1'] * sta['Y1'] + sta['A2'] * sta['Y2'])
        sta['Y2'] = sta['Y1']
        sta['Y1'] = OUTPUT
        sta['X2'] = sta['X1']
        sta['X1'] = sta['v']
        sta['v'] = OUTPUT
        # --------------End of Recursive Filter #

        # from ACC to VEL #
        acc = sta['v']
        sta['v'] = (acc + sta['acc0']) * sta['dt'] / 2.0 + sta['vel0']
        sta['acc0'] = acc
        sta['vel0'] = sta['v']
        sta['ave0'] = sta['ave0'] * (10. - sta['dt']) / 10.0 + sta['v'] * sta['dt'] / 10.0
        sta['v'] = sta['v'] - sta['ave0']

        # --------------For Recursive Filter  high pass 2 poles at 0.075 Hz #
        OUTPUT = sta['B0'] * sta['v'] + sta['B1'] * sta['XX1'] + sta['B2'] * sta['XX2']
        OUTPUT = OUTPUT - ( sta['A1'] * sta['YY1'] + sta['A2'] * sta['YY2'] )
        sta['YY2'] = sta['YY1']
        sta['YY1'] = OUTPUT
        sta['XX2'] = sta['XX1']
        sta['XX1'] = sta['v']
        sta['v'] = OUTPUT
        # --------------End of Recursive Filter #

        # --------------------------------------------------------------------------------------------
        sta['a'] = (sta['v'] - sta['avd1']) / sta['dt']

        acc_vel = sta['v']
        sta['d'] = (acc_vel + sta['acc0_vel']) * sta['dt'] / 2.0 + sta['vel0_dis']
        sta['acc0_vel'] = acc_vel
        sta['vel0_dis'] = sta['d']
        sta['ave1'] = sta['ave1'] * (10. - sta['dt']) / 10.0 + sta['d'] * sta['dt'] / 10.0
        sta['d'] = sta['d'] - sta['ave1']

        # --------------For Recursive Filter  high pass 2 poles at 0.075 Hz #
        OUTPUT = sta['B0'] * sta['d'] + sta['B1'] * sta['x1'] + sta['B2'] * sta['x2']
        OUTPUT = OUTPUT - ( sta['A1'] * sta['y1'] + sta['A2'] * sta['y2'] )
        sta['y2'] = sta['y1']
        sta['y1'] = OUTPUT
        sta['x2'] = sta['x1']
        sta['x1'] = sta['d']
        sta['d'] = OUTPUT
        # --------------End of Recursive Filter #
        sta['pa'] = max(sta['pa'], abs(sta['a']))
        sta['pv'] = max(sta['pv'], abs(sta['v']))
        sta['pd'] = max(sta['pd'], abs(sta['d']))

    return sta

