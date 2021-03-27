import sys
from os.path import join
import os
sys.path.insert(0,join(os.environ['HOME'], 'caffe', 'python'))
os.environ['LD_LIBRARY_PATH']=":".join([join(os.environ['HOME'],'miniconda3','lib'),"/opt/OpenBLAS/lib","/usr/local/lib"])
import caffe
from caffe import layers as L, params as P, to_proto

# some utility functions


def add_layer_to_net_spec(ns, caffe_layer, name, *args, **kwargs):
    kwargs.update({'name': name})
    l = caffe_layer(*args, **kwargs)
    ns.__setattr__(name, l)
    return ns.__getattr__(name)


def add_layer_with_multiple_tops(ns, caffe_layer, lname, ntop, *args, **kwargs):
    kwargs.update({'name': lname, 'ntop': ntop})
    num_in = len(args)-ntop  # number of input blobs
    tops = caffe_layer(*args[:num_in], **kwargs)
    for i in range(ntop):
        ns.__setattr__(args[num_in+i], tops[i])
    return tops

# implement single time step LSTM unit


def single_time_step_lstm(ns, h0, c0, x, prefix, num_output, weight_names=None):
    """
    see arXiv:1511.04119v1
    """
    if weight_names is None:
        weight_names = ['w_'+prefix+nm for nm in ['Mxw', 'Mxb', 'Mhw']]
    # full InnerProduct (incl. bias) for x input
    Mx = add_layer_to_net_spec(ns, L.InnerProduct, prefix+'lstm/Mx', x,
                               inner_product_param={'num_output': 4*num_output, 'axis': 2,
                                                    'weight_filler': {'type': 'uniform', 'min': -0.05, 'max': 0.05},
                                                    'bias_filler': {'type': 'constant', 'value': 0}},
                               param=[{'lr_mult': 1, 'decay_mult': 1, 'name': weight_names[0]},
                                      {'lr_mult': 2, 'decay_mult': 0, 'name': weight_names[1]}])
    Mh = add_layer_to_net_spec(ns, L.InnerProduct, prefix+'lstm/Mh', h0,
                               inner_product_param={'num_output': 4*num_output, 'axis': 2, 'bias_term': False,
                                                    'weight_filler': {'type': 'uniform', 'min': -0.05, 'max': 0.05},
                                                    'bias_filler': {'type': 'constant', 'value': 0}},
                               param={'lr_mult': 1, 'decay_mult': 1, 'name': weight_names[2]})
    M = add_layer_to_net_spec(ns, L.Eltwise, prefix+'lstm/Mx+Mh', Mx, Mh,
                              eltwise_param={'operation': P.Eltwise.SUM})
    raw_i1, raw_f1, raw_o1, raw_g1 = \
        add_layer_with_multiple_tops(ns, L.Slice, prefix+'lstm/slice', 4, M,
                                     prefix+'lstm/raw_i', prefix+'lstm/raw_f', prefix +
                                     'lstm/raw_o', prefix+'lstm/raw_g',
                                     slice_param={'axis': 2, 'slice_point': [num_output, 2*num_output, 3*num_output]})
    i1 = add_layer_to_net_spec(
        ns, L.Sigmoid, prefix+'lstm/i', raw_i1, in_place=True)
    f1 = add_layer_to_net_spec(
        ns, L.Sigmoid, prefix+'lstm/f', raw_f1, in_place=True)
    o1 = add_layer_to_net_spec(
        ns, L.Sigmoid, prefix+'lstm/o', raw_o1, in_place=True)
    g1 = add_layer_to_net_spec(
        ns, L.TanH, prefix+'lstm/g', raw_g1, in_place=True)
    c1_f = add_layer_to_net_spec(
        ns, L.Eltwise, prefix+'lstm/c_f', f1, c0, eltwise_param={'operation': P.Eltwise.PROD})
    c1_i = add_layer_to_net_spec(
        ns, L.Eltwise, prefix+'lstm/c_i', i1, g1, eltwise_param={'operation': P.Eltwise.PROD})
    c1 = add_layer_to_net_spec(ns, L.Eltwise, prefix+'lstm/c',
                               c1_f, c1_i, eltwise_param={'operation': P.Eltwise.SUM})
    # cannot override c - it MUST be preserved for next time step!!!
    act_c = add_layer_to_net_spec(
        ns, L.TanH, prefix+'lstm/act_c', c1, in_place=False)
    h1 = add_layer_to_net_spec(ns, L.Eltwise, prefix+'lstm/h',
                               o1, act_c, eltwise_param={'operation': P.Eltwise.PROD})
    return c1, h1, weight_names


def exmaple_use_of_lstm():
    T = 3  # number of time steps
    B = 10  # batch size
    lstm_output = 500  # dimension of LSTM unit

    # use net spec
    ns = caffe.NetSpec()

    # we need initial values for h and c
    ns.h0 = L.DummyData(name='h0', dummy_data_param={'shape': {'dim': [1, B, lstm_output]},
                                                     'data_filler': {'type': 'constant', 'value': 0}})

    ns.c0 = L.DummyData(name='c0', dummy_data_param={'shape': {'dim': [1, B, lstm_output]},
                                                     'data_filler': {'type': 'constant', 'value': 0}})

    # simulate input X over T time steps and B sequences (batch size)
    ns.X = L.DummyData(name='X', dummy_data_param={
                       'shape': {'dim': [T, B, 128, 10, 10]}})
    # slice X for T time steps
    xt = L.Slice(ns.X, name='slice_X', ntop=T, slice_param={
                 'axis': 0, 'slice_point': list(range(1, T))})
    # unroling
    h = ns.h0
    c = ns.c0
    lstm_weights = None
    tops = []
    for t in range(T):
        c, h, lstm_weights = single_time_step_lstm(
            ns, h, c, xt[t], 't'+str(t)+'/', lstm_output, lstm_weights)
        tops.append(h)
        ns.__setattr__('c'+str(t), c)
        ns.__setattr__('h'+str(t), h)
    # concat all LSTM tops (h[t]) to a single layer
    ns.H = L.Concat(*tops, name='concat_h', concat_param={'axis': 0})
    return ns


ns = exmaple_use_of_lstm()
with open('lstm_demo.prototxt', 'w') as W:
    W.write('name: "LSTM using NetSpec example"\n')
    W.write('%s\n' % ns.to_proto())
loss1[it] = mysolver.net.blobs['loss'].data.copy()

data = mysolver.net.blobs['data'].data.copy()
fc6 = mysolver.net.blobs['fc6'].data.copy()
label = mysolver.net.blobs['label'].data.copy()
clip_markers = mysolver.net.blobs['clip_markers'].data.copy()

fc6_reshape = mysolver.net.blobs['fc6-reshape'].data.copy()
reshape_label = mysolver.net.blobs['reshape-label'].data.copy()
reshape_cm = mysolver.net.blobs['reshape-cm'].data.copy()
lstm1 = mysolver.net.blobs['lstm1'].data.copy()
fc8_final = mysolver.net.blobs['fc8-final'].data.copy()

# 24 the number of video , train_buffer
# 16 the length of processed clip, train_frames

print('data: ', data.shape)
print('fc6: ', fc6.shape)
print('label: ', label.shape)
print('clip_markers: ', clip_markers.shape)

print('fc6_reshape: ', fc6_reshape.shape)
print('reshape_cm: ', reshape_cm.shape)
print('lstm1: ', lstm1.shape)

print('reshape_label: ', reshape_label.shape)
print('fc8-final: ', fc8_final.shape)

print('shape over...')
