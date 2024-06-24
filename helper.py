"""
Command Line reader
Author: Fejiro Aigboro + Hanna Abrahem
Date: May 15th
"""
import sys
import optparse

def parse_args():
    """Parse command line arguments ()."""
    parser = optparse.OptionParser(description='Command')
    
    cnns = "cnn1: CNN trained with 10pics per person\n\
            cnn2: CNN trained with all images"
    parser.add_option('-c', '--CNNModel', type='string', help=cnns, default="cnn1")
    parser.add_option('-t', '--train', type='string', help="True if you want to train model,\
      False otherwise", default="False")
    parser.add_option('-e', '--Epochs', type='int', help="# of epochs", default=500)

    (opts, args) = parser.parse_args()

    mandatories = ['train']
    for m in mandatories:
        if not opts.__dict__[m]:
            print('mandatory option ' + m + ' is missing\n')
            parser.print_help()
            sys.exit()
    train = ['True', 'False']
    if opts.train not in train: # ensure input is one of the data listed
      error = "The accepted options for train are: " + " ".join(train)
      print(error)
      sys.exit()
    models = ['cnn1', 'cnn2']
    if opts.CNNModel not in models: # ensure input is one of the data listed
      error = "The accepted options for CNN models are: " + " ".join(models)
      print(error)
      sys.exit()

    return opts