TRAINING THE SENTENCE-LEVEL MODEL:(Here the data files should be in format source ||| target)

./build_gpu/transformer-train --dynet_mem 15500 --minibatch-size 1000 --treport 7500 --dreport 37500 -t $trainfname -d $devfname --model-path $modelfname \
--sgd-trainer 4 --lr-eta 0.0001 -e 35 --patience 10 --use-label-smoothing --encoder-emb-dropout-p 0.1 --encoder-sublayer-dropout-p 0.1 --decoder-emb-dropout-p 0.1 \
--decoder-sublayer-dropout-p 0.1 --attention-dropout-p 0.1 --ff-dropout-p 0.1 --ff-activation-type 1 --nlayers 4 --num-units 512 --num-heads 8

Note: modelfname is the directory which contains the model files and vocabs specified by option model-path.

COMPUTING THE REPRESENTATIONS:(Manually make directories for representations in model-path src-rep or tgt-rep before running, Here the data files should be in format docid ||| source ||| target)

./build_gpu/transformer-computerep --dynet_mem 15500 --model-path $modelfname --input_doc $fname --input_type 2 --rep_type 1

(Note: rep_type tells whether the representation computed is monolingual (from encoder) or bilingual (from decoder) and input_type tells 
whether the input is training, dev or test.)

TRAINING THE DOCUMENT-LEVEL MODEL:(Here the data files should be in format docid ||| source ||| target)
Train a hierarchical sparse-soft model with monolingual context.

./build_gpu/transformer-context --dynet_mem 15500 --dynet-devices GPU:0,CPU --minibatch-size 1000 --dtreport 240 --ddreport 1200 --update-steps 5 \
--train_doc $trainfname --devel_doc $devfname --model-path $modelfname --model-file $modelname --context-type 1 --doc-attention-type 3 --use-sparse-soft 1 \
--use-new-dropout --encoder-emb-dropout-p 0.2 --encoder-sublayer-dropout-p 0.2 --decoder-emb-dropout-p 0.2 --decoder-sublayer-dropout-p 0.2 \
--attention-dropout-p 0.2 --ff-dropout-p 0.2 --sgd-trainer 4 --lr-eta 0.0001 -e 35 --patience 10 

"model-file" assigns the name to the model config and param file as given by modelname.
The option "dynet-devices" is only required when using hierarchical attention. 
"update-steps" is required when documents are shorter for e.g. news-commentary and Europarl. 
"context-type" is 1 for monolingual and 2 for bilingual. 
"doc-attention-type" is 1 for sentence-level, 2 for word-level and 3 for hierarchical. If set to 3 then may also need to set "use-sparse-soft" which is by default
set to sparse at sentence and soft attention at word-level.

DECODING THE SENTENCE-LEVEL MODEL:
./build_gpu/transformer-decode --dynet_mem 15000 --model-path $modelfname --beam 1 -T $testfname 

DECODING THE DOCUMENT-LEVEL MODEL:
./build_gpu/transformer-context-decode --dynet-mem 15000 --model-path $modelfname --model-file $modelname --beam 1 -T $testfname --context-type 1

trainfname, devfname and testfname are the paths to the respective data files in the required format.
