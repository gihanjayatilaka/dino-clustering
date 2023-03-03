print("Hi!")

import argparse

args = argparse.ArgumentParser()
args.add_argument("--outputFolder","-o",type=str,default="./out/")

