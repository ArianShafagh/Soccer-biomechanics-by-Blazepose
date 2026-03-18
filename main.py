from model import Blazepose
from Loader import Loader

def main():
    params = Loader().load()
    model = Blazepose(*params)
    model.run()

if __name__ == "__main__":
    main()
