#!/usr/bin/env python3
import wave
import numpy as np
import scipy as sp
from scipy.fft import fft
from scipy import signal
import matplotlib.pyplot as plt
#from commpy import rrcosfilter

varicode = {
    '1010101011' : '\x00',    '1011011011' : '\x01',
    '1011101101' : '\x02',    '1101110111' : '\x03',
    '1011101011' : '\x04',    '1101011111' : '\x05',
    '1011101111' : '\x06',    '1011111101' : '\x07',
    '1011111111' : '\x08',    '11101111'   : '\x09',
    '11101'      : '\x0A',    '1101101111' : '\x0B',
    '1011011101' : '\x0C',    '11111'      : '\x0D',
    '1101110101' : '\x0E',    '1110101011' : '\x0F',
    '1011110111' : '\x10',    '1011110101' : '\x11',
    '1110101101' : '\x12',    '1110101111' : '\x13',
    '1101011011' : '\x14',    '1101101011' : '\x15',
    '1101101101' : '\x16',    '1101010111' : '\x17',
    '1101111011' : '\x18',    '1101111101' : '\x19',
    '1110110111' : '\x1A',    '1101010101' : '\x1B',
    '1101011101' : '\x1C',    '1110111011' : '\x1D',
    '1011111011' : '\x1E',    '1101111111' : '\x1F',
    '1'          : ' ',       '111111111'  : '!',
    '101011111'  : '"',       '111110101'  : '#',
    '111011011'  : '$',       '1011010101' : '%',
    '1010111011' : '&',       '101111111'  : '\'',
    '11111011'   : '(',       '11110111'   : ')',
    '101101111'  : '*',       '111011111'  : '+',
    '1110101'    : ',',       '110101'     : '-',
    '1010111'    : '.',       '110101111'  : '/',
    '10110111'   : '0',       '10111101'   : '1',
    '11101101'   : '2',       '11111111'   : '3',
    '101110111'  : '4',       '101011011'  : '5',
    '101101011'  : '6',       '110101101'  : '7',
    '110101011'  : '8',       '110110111'  : '9',
    '11110101'   : ':',       '110111101'  : ';',
    '111101101'  : '<',       '1010101'    : '=',
    '111010111'  : '>',       '1010101111' : '?',
    '1010111101' : '@',       '1111101'    : 'A',
    '11101011'   : 'B',       '10101101'   : 'C',
    '10110101'   : 'D',       '1110111'    : 'E',
    '11011011'   : 'F',       '11111101'   : 'G',
    '101010101'  : 'H',       '1111111'    : 'I',
    '111111101'  : 'J',       '101111101'  : 'K',
    '11010111'   : 'L',       '10111011'   : 'M',
    '11011101'   : 'N',       '10101011'   : 'O',
    '11010101'   : 'P',       '111011101'  : 'Q',
    '10101111'   : 'R',       '1101111'    : 'S',
    '1101101'    : 'T',       '101010111'  : 'U',
    '110110101'  : 'V',       '101011101'  : 'W',
    '101110101'  : 'X',       '101111011'  : 'Y',
    '1010101101' : 'Z',       '1111111'    : 'I',
    '111111101'  : 'J',       '101111101'  : 'K',
    '11010111'   : 'L',       '10111011'   : 'M',
    '11011101'   : 'N',       '10101011'   : 'O',
    '11010101'   : 'P',       '111011101'  : 'Q',
    '10101111'   : 'R',       '1101111'    : 'S',
    '1101101'    : 'T',       '101010111'  : 'U',
    '110110101'  : 'V',       '101011101'  : 'W',
    '101110101'  : 'X',       '101111011'  : 'Y',
    '1010101101' : 'Z',       '111110111'  : '[',
    '111101111'  : '\\',      '111111011'  : ']',
    '1010111111' : '^',       '101101101'  : '_',
    '1011011111' : '`',       '1011'       : 'a',
    '1011111'    : 'b',       '101111'     : 'c',
    '101101'     : 'd',       '11'         : 'e',
    '111101'     : 'f',       '1011011'    : 'g',
    '101011'     : 'h',       '1101'       : 'i',
    '111101011'  : 'j',       '10111111'   : 'k',
    '11011'      : 'l',       '111011'     : 'm',
    '1111'       : 'n',       '111'        : 'o',
    '111111'     : 'p',       '110111111'  : 'q',
    '10101'      : 'r',       '10111'      : 's',
    '101'        : 't',       '110111'     : 'u',
    '1111011'    : 'v',       '1101011'    : 'w',
    '11011111'   : 'x',       '1011101'    : 'y',
    '111010101'  : 'z',       '1010110111' : '{',
    '110111011'  : '|',       '1010110101' : '}',
    '1011010111' : '~',       '1110110101' : '\x7F' }

with wave.open('transmission.wav') as file:
    n = file.getparams().nframes
    f = file.getparams().framerate
    print("The sampling frequency is: ",f)
    txbytes = file.readframes(n)
    tx = np.frombuffer(txbytes, dtype=np.int16)

    plt.plot(tx[0:500])
    plt.title("beginning of input signal")
    plt.show()

    plt.plot(tx[0:8000])
    plt.title("beginning of input signal, probably synchronization sequence")
    plt.show()

    plt.plot(range(10000,30000),tx[10000:30000])
    plt.title("middle of input signal")
    plt.show()

    plt.plot( np.arange(0,int(f/2),f/len(tx)), np.abs( fft(tx) )[0:int(len(tx)/2)])
    plt.title("fft of input signal")
    plt.show()

    # Create cos at 1.5kHz
    cos = np.array( [ np.cos( 2 * (k % 16) * int(15000000/8000) / 10000 * np.pi ) for k in range(0, len(tx)) ] )
    plt.plot( range(0,int(f/2)), np.abs( fft(cos[0:f]) )[0:int(f/2)] )
    plt.title("Plot of fft of generated cosine signal")
    plt.xlabel('Frequency')
    plt.show()

    dem = 2 * cos * tx
    plt.plot( range(0,int(f/2)), np.abs( fft(dem[0:f]) )[0:int(f/2)] )
    plt.title("Plot of FFT of demodulated signal")
    plt.xlabel('Frequency')
    plt.show()

    soslpf = signal.butter(10, 1500, btype='lowpass', output='sos', fs=f)
    freq,mag = signal.sosfreqz(soslpf, fs=f)
    plt.plot(freq,np.abs(mag))
    plt.title("Plot of Low-Pass Filter Frequency Response")
    plt.xlabel('Frequency')
    plt.show()

    demfilt = signal.sosfilt(soslpf, dem)
    plt.plot( range(0,int(f/2)), np.abs( fft( demfilt[0:f] ) )[0:int(f/2)] )
    plt.title("Plot of Signal After Low-Pass Filtering")
    plt.xlabel('Frequency')
    plt.show()

    plt.plot(demfilt)
    plt.stem( range(11,len(demfilt),256), [demfilt[i] for i in range(11,len(demfilt),256) ], 
        linefmt='C1-', markerfmt='C1o', use_line_collection=True )
    plt.title("Placing line markers every 256 samples after seeing periodic \nbehavior in the synchronization sequence in the beginning\nShould use some sort of symbol clock recovery")
    plt.show()

    data = np.array( [ np.sign( demfilt[i]) for i in range(11,len(demfilt),256) ] )
    data = data/2 + 0.5
    data = data.astype(int)

    diffdecode = np.array( [ data[i]^data[i-1]^1 for i in range(1, len(data) ) ] )
    plt.stem(diffdecode, use_line_collection=True)
    plt.title("Data after being differentially decoded")
    plt.show()

    varicoded_data = ''.join(map(str,diffdecode)).lstrip('0').split('00')

    flag = ''

    for bitstring in varicoded_data:
        if bitstring in set( varicode.keys() ):
            flag = flag + varicode[bitstring]

    print(flag)
