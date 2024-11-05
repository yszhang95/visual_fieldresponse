from abc import ABC, abstractmethod
import functools
import numpy as np
import scipy

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class PixelPlotter():
    def __init__(self):
        pass
    def plot_pixel(self, pixels, **kwargs):

        frames = kwargs.get('frames', None)

        if frames is None:
            frames = pixels.shape[2]

        vmin = np.min(pixels)
        vmax = np.max(pixels)

        d = vmax - vmin
        d10percent = 0.1 * d

        vmin = vmin - d10percent
        vmax = vmax + d10percent

        fig, ax = plt.subplots()

        img = ax.imshow(pixels[:,:,0], vmin=vmin, vmax=vmax,
                        cmap='rainbow')
        fig.colorbar(img)
        def update_img(frame):
            H = pixels[:,:,frame]
            ax.set_title('Time tick {}'.format(frame))
            img.set_data(H)
            return img
        ani = animation.FuncAnimation(fig=fig, func=update_img,
                                      frames=frames,
                                      interval=5) # 5 ms each frame
        oname = kwargs.get('oname', 'pixel.gif')
        fps = kwargs.get('fps', 50)
        ani.save(oname, fps=fps)

class Response(ABC):
    @abstractmethod
    def load_signal(self, signal):
        '''load data'''
        pass

    @abstractmethod
    def _load_response(self, response):
        pass

    @abstractmethod
    def output(self):
        pass


class FieldResponse(Response):
    def __init__(self):
        self.response = None
        self.signal = None

    def load_signal(self, signal):
        self.signal = signal

    def _load_response(self, response):
        self.response = response

    def load_response_npz(self, f, single_quadrant=True):
        r = np.load(f)
        rx, ry, rt = r.shape
        if single_quadrant:
            rnew = np.zeros([2*rx, 2*ry, rt])
            rnew[:rx, :ry, :] = r[::-1, ::-1, :]
            rnew[rx:, :ry, :] = r[:, ::-1, :]
            rnew[:rx, ry:, :] = r[::-1, :, :]
            rnew[rx:, ry:, :] = r[:, :, :]
        self._load_response(rnew)

    def output(self):
        result = scipy.signal.convolve(self.signal,
                                       self.response, mode='full')
        rx, ry, rt = self.response.shape
        sx, sy, st = self.signal.shape

        result = result[:,:, rt-1:]
        return result

class PixelResponse(Response):
    def __init__(self):
        pass
    def load_signal(self):
        pass
    def _load_response(self):
        pass
    def output(self):
        pass


class ElectronicResponse(Response):
    def __init__(self):
        pass
    def load_signal(self):
        pass
    def _load_response(self):
        pass
    def output(self):
        pass


class QShape():
    def __init__(self):
        self.N_PIXEL_X = 410 # 41 pixels
        self.N_PIXEL_Y = 410 # 41 pixels
        self.N_TIME = 2500 # 2500 time ticks
    def __zero_signal(self):
        return np.zeros([self.N_PIXEL_X, self.N_PIXEL_Y, self.N_TIME])

    def __shift_xyindices(self, indices):
        idx = (indices[1] + self.N_PIXEL_X // 2,
               indices[0] + self.N_PIXEL_Y // 2)
        return idx
    def point_charge(self, xpos=0, ypos=0, tpos=0, Q=1.):
        xidx, yidx = self.__shift_xyindices([xpos, ypos])
        tidx = tpos
        q = self.__zero_signal()
        q[xidx, yidx, tidx] = Q
        return q

    def horizontal_line(self, ypos=0, tpos=0, Q=1.):
        if isinstance(Q, np.ndarray):
            if Q.shape[0] != self.N_PIXEL_X or len(Q.shape)>1:
                raise NotImplementedError
        q = self.__zero_signal()
        xidx, _ = self.__shift_xyindices((0, ypos))
        q[xidx,:,tpos] = Q
        return q

if __name__ == '__main__':
    qshape = QShape()
    # tpos = 1999, meaning the start of field response in the example
    # small tpos means closer to anode
    point_q_pxls = qshape.point_charge(tpos=1999)
    hline_q_pxls = qshape.horizontal_line()

    # plotter.plot_pixel(point_q_pxls, oname='point_q.gif')
    # plotter.plot_pixel(hline_q_pxls, oname='hline_q.gif')

    fr = FieldResponse()
    fr.load_response_npz('response_44_v2a_100ns.npy')

    fr.load_signal(point_q_pxls)

    point_q_r = fr.output()


    plotter = PixelPlotter()
    plotter.plot_pixel(fr.response, oname='fr.gif',
                       frames=range(1700, 2000))
    plotter.plot_pixel(point_q_r, oname='point_q_r.gif',
                       frames=range(1700, 2500))

    # print(type(fr.response))
    # print(np.argwhere[fr.response>1])
    # print(point_q_r[point_q_r>1])
    # print(fr.response[fr.response>1])
    # print(np.argwhere(point_q_r>1))
