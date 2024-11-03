import spacy
import math
import sys
from collections import defaultdict

nlp = spacy.load("pt_core_news_lg")

def lematizar_e_filtrar(texto):
    doc = nlp(texto.lower())
    return [token.lemma_.lower() for token in doc if not token.is_stop and (token.is_alpha or token.like_num or token.lemma_ == "2x" ) and not " " in token.lemma_]

def construir_indice(base_docs):
    indice = defaultdict(lambda: defaultdict(int))
    for doc_id, doc_text in enumerate(base_docs, 1):
        termos = lematizar_e_filtrar(doc_text)
        for termo in termos:
            indice[termo][doc_id] += 1
    return indice

def calcular_tfidf(indice, num_docs):
    tfidf = defaultdict(lambda: defaultdict(float))
    for termo, docs in indice.items():
        df = len(docs)  
        idf = math.log10(num_docs / df) if df > 0 else 0
        for doc_id, freq in docs.items():
            tf = 1 + math.log10(freq) if freq > 0 else 0
            tfidf[doc_id][termo] = tf * idf
    return tfidf

def salvar_indice(indice, caminho="indice.txt"):
    with open(caminho, "w", encoding="utf-8") as f:
        for termo, docs in sorted(indice.items()):
            linha = f"{termo}: " + " ".join([f"{doc_id},{freq}" for doc_id, freq in sorted(docs.items())])
            f.write(linha + "\n")

def salvar_pesos(tfidf, doc_paths, caminho="pesos.txt"):
    with open(caminho, "w", encoding="utf-8") as f:
        for doc_id, termos in sorted(tfidf.items()):
            linha_termos = []
            for termo, peso in sorted(termos.items()):
                if peso > 0:
                    linha_termos.append(f"{termo},{peso}")
            linha = f"{doc_paths[doc_id - 1]}: " + " ".join(linha_termos)
            f.write(linha + "\n")

def ler_arquivo_com_fallback(caminho):
    encodings = ["utf-8", "latin-1", "utf-8-sig"]
    for encoding in encodings:
        try:
            with open(caminho, "r", encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"Não foi possível decodificar o arquivo {caminho} com os encodings tentados.")

def main():
    if len(sys.argv) < 2:
        print("Uso: python tfidf.py <caminho_base.txt>")
        sys.exit(1)

    base_path = sys.argv[1]
    doc_paths = ler_arquivo_com_fallback(base_path).strip().splitlines()
    base_docs = [ler_arquivo_com_fallback(path) for path in doc_paths]

    indice = construir_indice(base_docs)
    salvar_indice(indice)

    num_docs = len(base_docs)
    tfidf = calcular_tfidf(indice, num_docs)
    salvar_pesos(tfidf, doc_paths)

if __name__ == "__main__":
    main()
