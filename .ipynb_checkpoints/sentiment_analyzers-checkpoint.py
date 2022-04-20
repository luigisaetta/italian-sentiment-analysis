import torch
from torch import nn

# HuggingFace transformers (availale in OCI DS conda nlp env)
# see: https://github.com/huggingface/transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification

#
# a more flexible version
#
class MultiSentimentAnalyzer:
    # load the tokenizer and transformer
    def __init__(self, MODEL_NAME, labels):

        #
        # attribute definitions
        #

        # for rounding the scores
        self._DEC_DIGITS = 4

        # name of HuggingFace model used
        self._MODEL_NAME = MODEL_NAME

        # the list of defined labels (and therefore we will have 5 scores)
        # this is changed from ITA class
        self._LABELS = labels

        print("Loading model...")
        self.tokenizer = AutoTokenizer.from_pretrained(self._MODEL_NAME)
        # Load the model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self._MODEL_NAME
        )

        print("Model loading completed!")

    #
    # does the scoring on a single sentence a time
    #
    def score(self, input_sentence):
        # encode the sentence and create the input tensor (in PyTorch format)
        input_ids = self.tokenizer(
            input_sentence, add_special_tokens=True, return_tensors="pt"
        )["input_ids"]

        # output from tokenizer is already a tensor

        # Call the model and get the logits
        with torch.no_grad():
            logits = self.model(input_ids)["logits"]

        # The model was trained with a Log Likelyhood + Softmax combined loss, hence to extract probabilities we need a softmax on top of the logits tensor
        proba = nn.functional.softmax(logits, dim=1)

        # to remove the added dimension with squeeze
        # proba is a tuple () with one value for each label
        scores = proba.squeeze(0)

        # get rid of tensor and round and
        # prepare the output json

        ret_vet = []

        for i, label in enumerate(self._LABELS):
            ret_vet.append({"label": label, "score": self.round(scores[i])})

        return ret_vet
    
    #
    # works on a list of sentences, with a single call to model
    #
    def batch_score(self, input_sentences):
        # encode the sentence and create the input tensor (in PyTorch format)
        # requires padding
        tokens = self.tokenizer(
            input_sentences, add_special_tokens=True, padding=True,
            return_tensors="pt"
        )

        # output from tokenizer is already a tensor
        
        # Call the model and get the logits
        # in batch scoring I need also to pass the attention mask
        with torch.no_grad():
            logits = self.model(**tokens)['logits']
        
        proba = nn.functional.softmax(logits, dim=1)
        
        return proba
    
    # only to format output
    def format_scores(self, scores):
        score_str = ""

        for v_score in scores:
            score_str += str(v_score["label"]) + ": " + str(v_score["score"]) + "|"

        return score_str

    # utility to get rid of tensor and round
    def round(self, tens_val):
        return round(tens_val.item(), self._DEC_DIGITS)
    
    