class App:
    def __init__(self, nc):

        self.nc = nc
        self.pg = False
        self.arg = {}

        self.bcm = None
        self.bcs = None

        self.finvv = None
        self.finvi = None
        self.finvb = None
        self.finvv = None
        self.fvisi = None
        self.fvisb = None
        self.fvisv = None
        self.fvisub = None
        self.src = None