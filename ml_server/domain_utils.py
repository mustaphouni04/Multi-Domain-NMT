from datasets import load_dataset
from typing import List, Dict, Union, Optional
from pathlib import Path
import lxml.etree as etree
import pandas as pd
from tqdm import tqdm
from euro_voc import EURO_VOC_DOMAINS
import numpy as np 
from numpy import dot
from numpy.linalg import norm 
from sentence_transformers import SentenceTransformer 

class DomainProcessing:
    def __init__(self, paths: Union[str,Path, List[Union[str, Path]]], name: Optional[str] = None):
        self.paths = [Path(p) for p in (paths if isinstance(paths,list) else [paths])]
        self.domain = None
        self.sentences = []
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.domain_embeddings = self.model.encode(EURO_VOC_DOMAINS, normalize_embeddings = True)

        if str(self.paths[0]).endswith(".csv") and isinstance(str(self.paths[0]), str):
            self.domain = pd.read_csv(self.paths[0])
        else:
            for path in self.paths:
                try:
                    self.domain = load_dataset(str(path), name=name, split="train")
                    self.content = None
                except:
                    if type(path) == Path:
                        path = str(path)
                    else:
                        try:
                            # Load a TMX file
                            tmx_file: etree._ElementTree = etree.parse(
                                    str(path), etree.XMLParser(encoding="utf-16le")
                            )
                            tmx_root: etree._Element = tmx_file.getroot()
                            self.sentences.extend(self.extract_translations(tmx_root))
                        except etree.XMLSyntaxError as e:
                            raise ValueError(f"Failed to parse TMX file: {e}")
                        except Exception as e:
                            raise RuntimeError(f"Unexpected error while processing TMX file: {e}")

    def extract_translations(self, root: etree._Element) -> List[Dict[str,str]]:
        translations = []

        for tu in tqdm(root.findall(".//tu")):
            en_sentence = None
            es_sentence = None

            for tuv in tu.findall(".//tuv"):
                lang = tuv.get("lang")
                seg = tuv.find("seg")

                if seg is not None and lang is not None:
                    if lang.startswith("EN-GB"):
                        en_sentence = seg.text.strip() if seg.text else ""
                    elif lang.startswith("ES-ES"):
                        es_sentence = seg.text.strip() if seg.text else ""

            segment_embedding = self.model.encode(en_sentence, normalize_embeddings=True)
            similarities = self.domain_embeddings @ segment_embedding.T
            best_index = np.argmax(similarities)
            best_domain = EURO_VOC_DOMAINS[best_index]

            if en_sentence and es_sentence:
                translations.append({"source":en_sentence, "target":es_sentence, "domain": best_domain})
        return translations

    @property
    def type(self):
        if self.domain is not None:
            return type(self.domain)
        else:
            return type(self.sentences)
    
    def process(self):
        if self.sentences:
            return pd.DataFrame(self.sentences)
        return self.domain





