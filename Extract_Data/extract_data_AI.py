import os
import glob
import random
import numpy as np
import data_info as di

import struct

# ----------------------------------------------------------------------------------------------------------------------
# To configure: two variables
#YUV_PATH_ORI = '/media/F/YUV_All/' # path storing 12 raw YUV files
#INFO_PATH = '/media/F/DataHEVC/AI_Info/' # path storing Info_XX.dat files for All-Intra configuration

#YUV_PATH_ORI = '/home/kkheon/test_images/ClassB/' # path storing 12 raw YUV files

YUV_PATH_ORI = '/data/kkheon/test_images/CPIH/' # path storing 12 raw YUV files
INFO_PATH = '/data/kkheon/test_images/CPIH/AI_Info/' # path storing Info_XX.dat files for All-Intra configuration
# ----------------------------------------------------------------------------------------------------------------------

YUV_NAME_LIST_FULL = di.YUV_NAME_LIST_FULL
YUV_WIDTH_LIST_FULL = di.YUV_WIDTH_LIST_FULL
YUV_HEIGHT_LIST_FULL = di.YUV_HEIGHT_LIST_FULL

#QP_LIST = [22, 27, 32, 37]
QP_LIST = [22]

N_QP_SPACE = 52

#INDEX_LIST_TRAIN = list(range(0, 4))
#INDEX_LIST_VALID = list(range(4, 8))
#INDEX_LIST_TEST = list(range(8, 12))

INDEX_LIST_TRAIN = list(range(0, 1))

def get_file_list(yuv_path_ori, info_path, yuv_name_list_full, qp_list, index_list):

    yuv_name_list = [yuv_name_list_full[index] for index in index_list]
    yuv_file_list = []
    info_file_list = []
    info_file_list_rd = []

    for i_qp in range(len(qp_list)):
        info_file_list.append([])
        info_file_list_rd.append([])
        for i_seq in range(len(index_list)):
            info_file_list[i_qp].append([])
            info_file_list_rd[i_qp].append([])

    for i_seq in range(len(index_list)):
            yuv_file_list.append([])

    for i_seq in range(len(index_list)):
        yuv_file_temp = glob.glob(yuv_path_ori + yuv_name_list[i_seq] + '.yuv')
        assert(len(yuv_file_temp) == 1)
        yuv_file_list[i_seq] = yuv_file_temp[0]

        for i_qp in range(len(qp_list)):
            #info_file_temp = glob.glob(info_path + 'Info*_' + yuv_name_list[i_seq] + '_*qp' + str(qp_list[i_qp]) + '*CUDepth.dat')
            info_file_temp = glob.glob(info_path + 'Info*_' + yuv_name_list[i_seq] + '*CUDepth.dat')
            assert(len(info_file_temp) == 1)
            info_file_list[i_qp][i_seq] = info_file_temp[0]

        for i_qp in range(len(qp_list)):
            #info_file_temp = glob.glob(info_path + 'Info*_' + yuv_name_list[i_seq] + '_*qp' + str(qp_list[i_qp]) + '*RD.dat')
            info_file_temp = glob.glob(info_path + 'Info*_' + yuv_name_list[i_seq] + '*RD.dat')
            assert(len(info_file_temp) == 1)
            info_file_list_rd[i_qp][i_seq] = info_file_temp[0]

    return yuv_file_list, info_file_list, info_file_list_rd

class FrameYUV(object):
    def __init__(self, Y, U, V):
        self._Y = Y
        self._U = U
        self._V = V

def get_file_size(path):
    try:
        size = os.path.getsize(path)
        return size
    except Exception as err:
        print(err)

def get_num_YUV420_frame(file, width, height):
    file_bytes = get_file_size(file)
    frame_bytes = width * height * 3 // 2
    assert(file_bytes % frame_bytes == 0)
    return file_bytes // frame_bytes

def read_YUV420_frame(fid, width, height):
    # read a frame from a YUV420-formatted sequence
    d00 = height // 2
    d01 = width // 2
    Y_buf = fid.read(width * height)
    Y = np.reshape(np.frombuffer(Y_buf, dtype=np.uint8), [height, width])
    U_buf = fid.read(d01 * d00)
    U = np.reshape(np.frombuffer(U_buf, dtype=np.uint8), [d00, d01])
    V_buf = fid.read(d01 * d00)
    V = np.reshape(np.frombuffer(V_buf, dtype=np.uint8), [d00, d01])
    return FrameYUV(Y, U, V)

def read_info_frame(fid, width, height, mode):
    # read information of CU/TU partition
    assert(width % 8 == 0 and height % 8 == 0)
    if mode == 'CU':
        unit_width = 16
    elif mode == 'TU':
        unit_width = 8
    num_line_in_unit = height // unit_width
    num_column_in_unit = width // unit_width

    info_buf = fid.read(num_line_in_unit * num_column_in_unit)
    info = np.reshape(np.frombuffer(info_buf, dtype=np.uint8), [num_line_in_unit, num_column_in_unit])
    return info

def read_rd_frame(fid, width, height, unit_width=64, num_value=21):

    num_line_in_unit = height // unit_width
    num_column_in_unit = width // unit_width

    read_byte_size = num_line_in_unit * num_column_in_unit * num_value * np.dtype(np.float64).itemsize
    info_buf = fid.read(read_byte_size)

    info = np.reshape(np.frombuffer(info_buf, dtype=np.float64), [num_line_in_unit * num_column_in_unit, num_value])
    return info

def write_data(fid_out, frame_Y, qp_list, cu_depth_mat_list, cu_rd_mat_list, fid_out_log):
    width = np.shape(frame_Y)[1]
    height = np.shape(frame_Y)[0]
    assert(len(qp_list) == len(cu_depth_mat_list))
    assert(len(qp_list) == len(cu_rd_mat_list))

    n_qp = len(qp_list)
    n_line = height // 64
    n_col = width // 64
    for i_line in range(n_line):
        for i_col in range(n_col):
            buf_sample = (np.ones((4096 + 64 + 16 * N_QP_SPACE,)) * 255).astype(np.uint8)
            patch_Y = frame_Y[i_line*64: (i_line + 1)*64, i_col*64: (i_col + 1)*64]
            buf_sample[0:4096] = np.reshape(patch_Y, (4096,))

            for i_qp in range(n_qp):
                patch_cu_depth = cu_depth_mat_list[i_qp][i_line * 4: (i_line + 1) * 4, i_col * 4: (i_col + 1) * 4]
                i_start_in_buf = 4096 + 64 + qp_list[i_qp] * 16
                buf_sample[i_start_in_buf: i_start_in_buf + 16] = np.reshape(patch_cu_depth, (16,))
            fid_out.write(buf_sample)

            # write RD data
            buf_sample_rd = (np.zeros(21)).astype(np.float64)    # 21 values, 8 byte

            cu_index = n_col * i_line + i_col
            patch_cu_rd = cu_rd_mat_list[i_qp][cu_index:cu_index+1, :]
            #fid_out.write(patch_cu_rd)
            buf_sample_rd[:] = patch_cu_rd[:]
            fid_out.write(buf_sample_rd)

            # log
            fid_out_log.write(('CU=' + str(cu_index) + ' '))
            fid_out_log.write('\n')

            #fid_out_log.write(buf_sample)
            buf_sample[0:4096].tofile(fid_out_log, sep=' ', format='%3d')
            fid_out_log.write('\n')

            buf_sample[4096 + 64:].tofile(fid_out_log, sep=' ', format='%3d')
            fid_out_log.write('\n')

            #fid_out_log.write('RD info= ')
            #fid_out_log.write(patch_cu_rd)
            #patch_cu_rd.tofile(fid_out_log, sep=' ', format='%f')
            buf_sample_rd.tofile(fid_out_log, sep=' ', format='%f')
            fid_out_log.write('\n')

    return n_line * n_col

def shuffle_samples(file, sample_length):
    file_bytes = get_file_size(file)
    assert(file_bytes % sample_length == 0)
    num_samples = file_bytes // sample_length
    # random
    index_list = random.sample(range(num_samples), num_samples)
    ## incremental order (test)
    #index_list = range(num_samples)

    fid_in = open(file, 'rb')
    fid_out = open(file + '_shuffled', 'wb')
    fid_out_log = open(file + '_shuffled.log', 'w')

    for i in range(num_samples):
        fid_in.seek(index_list[i] * sample_length, 0)
        info_buf = fid_in.read(sample_length)
        fid_out.write(info_buf)

        if (i + 1) % 100 == 0:
            print('%s : %d / %d samples completed.' % (file, i + 1, num_samples))

        # log
        fid_out_log.write('CU=%d\n' % index_list[i])

        pixel = struct.unpack(len(info_buf[0:4096]) * "B", info_buf[0:4096])
        pixel = np.asarray(pixel)
        pixel.tofile(fid_out_log, sep=' ', format='%3d')
        fid_out_log.write('\n')

        #fid_out_log.write('depth_map=\n')
        cu_depth = struct.unpack(len(info_buf[4096+64:4096+64+16*52]) * "B", info_buf[4096+64:4096+64+16*52])
        cu_depth = np.asarray(cu_depth)
        cu_depth.tofile(fid_out_log, sep=' ', format='%3d')
        fid_out_log.write('\n')

        #fid_out_log.write('rd=\n')
        rd = struct.unpack(21 * "d", info_buf[4096+64+16*52:4096+64+16*52+21*8])
        rd = np.asarray(rd)
        rd.tofile(fid_out_log, sep=' ', format='%f')

        fid_out_log.write('\n')

    fid_in.close()
    fid_out.close()
    fid_out_log.close()

def generate_data(yuv_path_ori, info_path, yuv_name_list_full,
                  yuv_width_list_full, yuv_height_list_full, qp_list, index_list, save_file):

    yuv_file_list, info_file_list, info_file_list_rd = get_file_list(
        yuv_path_ori, info_path, yuv_name_list_full, qp_list, index_list)

    yuv_width_list = yuv_width_list_full[index_list]
    yuv_height_list = yuv_height_list_full[index_list]

    n_seq = len(yuv_file_list)
    n_qp = len(qp_list)

    fid_out = open(save_file, 'wb+')
    fid_out_log = open(save_file + '.log', 'w+')

    n_sample = 0
    for i_seq in range(n_seq):
        width = yuv_width_list[i_seq]
        height = yuv_height_list[i_seq]
        n_frame = get_num_YUV420_frame(yuv_file_list[i_seq], width, height)

        # temp
        n_frame = 2

        fid_yuv = open(yuv_file_list[i_seq], 'rb')
        fid_info_list = []
        fid_info_list_rd = []
        for i_qp in range(n_qp):
            fid_info_list.append(open(info_file_list[i_qp][i_seq], 'rb'))
            fid_info_list_rd.append(open(info_file_list_rd[i_qp][i_seq], 'rb'))

        for i_frame in range(n_frame):
            frame_YUV = read_YUV420_frame(fid_yuv, width, height)
            frame_Y = frame_YUV._Y
            cu_depth_mat_list = []
            cu_rd_mat_list = []
            for i_qp in range(n_qp):
                cu_depth_mat_list.append(read_info_frame(fid_info_list[i_qp], width, height, 'CU'))

                cu_rd_mat_list.append(read_rd_frame(fid_info_list_rd[i_qp], width, height, 64, 21))

            n_sample_one_frame = write_data(fid_out, frame_Y, qp_list, cu_depth_mat_list, cu_rd_mat_list, fid_out_log)
            n_sample += n_sample_one_frame
            print('Seq. %d / %d, %50s : frame %d / %d, %8d samples generated.' % (i_seq + 1, n_seq, yuv_file_list[i_seq], i_frame + 1, n_frame, n_sample))

        fid_yuv.close()
        for i_qp in range(n_qp):
            fid_info_list[i_qp].close()

    fid_out.close()
    fid_out_log.close()

    save_file_renamed = 'AI_%s_%d.dat' % (save_file, n_sample)
    os.rename(save_file, save_file_renamed)
    sample_unit_size = 4096 + 64 + 16 * N_QP_SPACE  #=4992
    #shuffle_samples(save_file_renamed, 4992)

    sample_unit_size += (21 * np.dtype(np.float64).itemsize)  # 4992 + 21*8=5160
    shuffle_samples(save_file_renamed, sample_unit_size)    # with RD value

if __name__ == '__main__':

    generate_data(YUV_PATH_ORI, INFO_PATH, YUV_NAME_LIST_FULL,
                  YUV_WIDTH_LIST_FULL, YUV_HEIGHT_LIST_FULL, QP_LIST, INDEX_LIST_TRAIN, 'Train')

    #generate_data(YUV_PATH_ORI, INFO_PATH, YUV_NAME_LIST_FULL,
    #              YUV_WIDTH_LIST_FULL, YUV_HEIGHT_LIST_FULL, QP_LIST, INDEX_LIST_VALID, 'Valid')

    #generate_data(YUV_PATH_ORI, INFO_PATH, YUV_NAME_LIST_FULL,
    #              YUV_WIDTH_LIST_FULL, YUV_HEIGHT_LIST_FULL, QP_LIST, INDEX_LIST_TEST, 'Test')
