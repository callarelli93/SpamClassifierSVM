import pip

def install(package):
    pip.main(['install', package])

install('email')
install('nltk')
install('sklearn')
