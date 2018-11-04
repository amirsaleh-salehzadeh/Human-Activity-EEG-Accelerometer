import argparse

from pythonosc import dispatcher
from pythonosc import osc_server
import csv
from pylab import *
from mpl_toolkits.mplot3d import Axes3D, axes3d
from threading import Thread, Event
import threading

isRecording = [False]
signalArrays = [0, 0, 0, 0, 0]
signalArraysArr = []
windowSize = 400

def runRecording():
    if isRecording[0]:
        with open("C:/RecordingFiles/data.csv", 'a', newline='') as logfile:
            writer = csv.writer(logfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(signalArrays)
            signalArraysArr.append(signalArrays)
    threading.Timer(0.007, runRecording).start()


def acc_handler(unused_addr, args, acc_x, acc_y, acc_z):
    signalArrays[2], signalArrays[3], signalArrays[4] = acc_x, acc_y, acc_z

    
def hs_handler(unused_addr, args, hs1, hs2, hs3, hs4):
    if hs2 == 1.0 and hs3 == 1.0:
        isRecording[0] = True
    else:
        isRecording[0] = False
    print (hs1, hs2, hs3, hs4)


def eeg_handler(unused_addr, args, l_ear, EEG1, EEG2, r_ear):
    signalArrays[0], signalArrays[1] = EEG1, EEG2


N = 3
cols = 3
rows = 1


def Gen_RandLine(length, dims=2):
    lineData = np.empty((dims, length))
    lineData[:, 0] = np.random.rand(dims)
    for index in range(1, length):
        step = abs((np.random.rand(dims) - 0.5) * 10)
        lineData[:, index] = lineData[:, index - 1] + step

    return lineData


def update_lines(num, dataLines, lines):
    for line, data in zip(lines, dataLines):
        line.set_data(data[0:2, :num])
        line.set_3d_properties(data[2, :num])
    return lines

# def recording():
#     if isRecording[0]:
#         with open("C:/RecordingFiles/data.csv", 'a', newline='') as logfile:
#             writer = csv.writer(logfile, delimiter=',',
#                             quotechar='|', quoting=csv.QUOTE_MINIMAL)
#             writer.writerow(signalArrays)
#             signalArraysArr.append(signalArrays)
#             time.sleep(5)
#     recording()
            
if __name__ == "__main__":
    isRecording = [False]
    signalArrays = [0, 0, 0, 0, 0]
#     gs = gridspec.GridSpec(rows, cols)
#     fig = plt.figure()
#     for n in range(N):
#         ax = fig.add_subplot(gs[n])
#         draw_muse_plot(ax, n)
#     fig.tight_layout()
#     plt.show()
#     data = [Gen_RandLine(10, 3) for index in range(5)]
#     lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1])[0] for dat in data]
#     animation.FuncAnimation(fig, update_lines, 25, fargs=(data, lines),
#                                        interval=50, blit=False)
    
#     gs = gridspec.GridSpec(rows, cols)
#     fig = plt.figure()
#     for n in range(N):
#         ax = fig.add_subplot(gs[n])
#         draw_muse_plot(ax)
#     fig.tight_layout()
#     fig = plt.figure()
#     ax = axes3d.Axes3D(fig)
#     data = [Gen_RandLine(10, 3) for index in range(5)]
#     lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1])[0] for dat in data]
#     ax.set_xlim3d([0.0, windowSize])
#     ax.set_xlabel('X')
#     ax.set_ylim3d([0.0, 5])
#     ax.set_ylabel('Y')
#     ax.set_zlim3d([0.0, 1600.0])
#     ax.set_zlabel('Z')
#     ax.set_title('3D Test')
#     line_ani = animation.FuncAnimation(fig, update_lines, 25, fargs=(data, lines),
#                                        interval=50, blit=False)
#     plt.show()
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip",
                        default="localhost",
                        help="The ip to listen on")
    parser.add_argument("--port",
                        type=int,
                        default=5003,
                        help="The port to listen on")
    args = parser.parse_args()
 
    dispatcher = dispatcher.Dispatcher()
    dispatcher.map("/debug", print)
    dispatcher.map("/muse/eeg", eeg_handler, "EEG")
    dispatcher.map("/muse/acc", acc_handler, "ACC")
    dispatcher.map("/muse/elements/horseshoe", hs_handler, "HSH")
    runRecording()
    server = osc_server.ThreadingOSCUDPServer(
        (args.ip, args.port), dispatcher)
    print("Serving on {}".format(server.server_address))
    server.serve_forever()
    
