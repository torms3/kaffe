import ConfigParser

class FakeSectionHead(object):

    def __init__(self, fp, section):
        self.fp = fp
        self.section = '[%s]\n' % section

    def readline(self):
        if self.section:
            try:
                return self.section
            finally:
                self.section = None
        else:
            return self.fp.readline()


class SolverParser(object):

    def parse(self, fname):
        p = ConfigParser.ConfigParser()
        p.readfp(FakeSectionHead(open(fname),'solver'))
        return p
