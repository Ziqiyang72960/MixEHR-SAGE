import logging
import argparse
import os
import torch
from MixEHR_SAGE import MixEHR_SAGE
from corpus import Corpus

logger = logging.getLogger("MixEHR-SAGE training processing")
parser = argparse.ArgumentParser(description="Train MixEHR-SAGE model")
# default arguments
parser.add_argument('corpus', help='Path to read corpus file', default='./store/')
parser.add_argument('output', help='Directory to store model', default='./result/')
parser.add_argument("-epoch", "--max_epoch", help="Maximum number of max_epochs", type=int, default=5)
parser.add_argument("-batch_size", "--batch_size", help="Batch size of a minibatch", type=int, default=1000)
parser.add_argument("-every", "--save_every", help="Store model every X number of iterations", type=int, default=1)
parser.add_argument("-seed_matrix", "--seed_matrix", help="Path to seed topic matrix", default="./phecode_mapping/seed_topic_matrix.pt")
parser.add_argument("-guide_prior", "--guide_prior_path", help="Path to guide prior directory", default="./guide_prior/")
parser.add_argument("-enable_temporal", "--enable_temporal", help="Enable temporal Markov chain inference", action='store_true')
parser.add_argument("-num_time_steps", "--num_time_steps", help="Number of time steps for temporal modeling", type=int, default=10)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cuda") # we use GPU, printed result is "cuda"
print(device)

def run(args):
    # print(args)
    # cmd = args.cmd
    seeds_topic_matrix = torch.load(args.seed_matrix, map_location=device, weights_only=False) # get seed word-topic mapping, V x K matrix
    print("V and K are", seeds_topic_matrix.shape) # torch.Size([V, K])
    corpus = Corpus.read_corpus_from_directory(args.corpus)
    print("trained modalities include", corpus.modalities)
    print(f"Number of modalities: {len(corpus.modalities)}")
    
    # Initialize model with temporal parameters
    model = MixEHR_SAGE(
        corpus, 
        seeds_topic_matrix, 
        corpus.modalities, 
        guided_modality=0, 
        stochastic_VI=True, 
        batch_size=args.batch_size, 
        out=args.output, 
        guide_prior_path=args.guide_prior_path,
        enable_temporal=args.enable_temporal,
        num_time_steps=args.num_time_steps
    )
    model = model.to(device)
    
    # Log temporal status
    if args.enable_temporal:
        logger.info(f"Temporal inference enabled with {args.num_time_steps} time steps")
    
    logger.info('''
    #     ======= Parameters =======
    #     mode: \t\ttraining
    #     file:\t\t%s
    #     output:\t\t%s
    #     max iterations:\t%s
    #     batch size:\t%s
    #     save every:\t\t%s
    #     temporal:\t\t%s
    #     ==========================
    # ''' % (args.corpus, args.output, args.max_epoch, args.batch_size, args.save_every, args.enable_temporal))
    elbo = model.inference(max_epoch=args.max_epoch, save_every=args.save_every)
    
    # Save temporal theta if temporal inference is enabled
    if args.enable_temporal:
        temporal_theta_path = os.path.join(args.output, 'temporal_theta.pt')
        model.save_temporal_theta(temporal_theta_path)
        logger.info(f"Saved temporal theta to {temporal_theta_path}")
    
    # Open files for writing
    with open('elbo1.txt', 'a') as file_elbo1, \
         open('elbo2.txt', 'a') as file_elbo2, \
         open('pz.txt', 'a') as file_pz, \
         open('qz.txt', 'a') as file_qz, \
         open('pw.txt', 'a') as file_pw:

        # Convert lists to strings and write to files
        file_elbo1.write(str(elbo) + '\n')
        file_elbo2.write(str(model.elbo) + '\n')
        file_pz.write(str(model.term1) + '\n')
        file_qz.write(str(model.term2) + '\n')
        file_pw.write(str(model.term3) + '\n')

if __name__ == '__main__':
    args = parser.parse_args()
    # If no arguments provided, use defaults for backward compatibility
    if args.corpus == './store/' and args.output == './result/' and len(__import__('sys').argv) == 1:
        run(parser.parse_args(['./store/', './result/']))
    else:
        run(args)
