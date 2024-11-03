import spacy
import sys
from collections import defaultdict

# Carregar o modelo Spacy para português
nlp = spacy.load("pt_core_news_lg")

def lematizar_e_filtrar(texto):
    # Converter todas as palavras para minúsculas e filtrar stopwords
    doc = nlp(texto.lower())
    return [token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha]

def construir_indice(base_docs):
    indice = defaultdict(lambda: defaultdict(int))
    for doc_id, doc_text in enumerate(base_docs, 1):
        termos = lematizar_e_filtrar(doc_text)
        for termo in termos:
            indice[termo][doc_id] += 1
    return indice

def salvar_indice(indice, caminho="indice.txt"):
    with open(caminho, "w") as f:
        for termo, docs in sorted(indice.items()):
            linha = f"{termo}: " + " ".join([f"{doc_id},{freq}" for doc_id, freq in docs.items()])
            f.write(linha + "\n")

def processar_consulta(consulta, indice):
    termos_consulta = consulta.split()
    resultado = set()

    for termo in termos_consulta:
        if termo in indice:
            docs_com_termo = set(indice[termo].keys())
            resultado.update(docs_com_termo)

    return resultado

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
    base_path = sys.argv[1]
    consulta_path = sys.argv[2]

    # Leitura da base de documentos com tentativa de vários encodings
    doc_paths = ler_arquivo_com_fallback(base_path).strip().splitlines()

    base_docs = []
    for path in doc_paths:
        base_docs.append(ler_arquivo_com_fallback(path))

    indice = construir_indice(base_docs)
    salvar_indice(indice)

    # Leitura e processamento da consulta com tentativa de vários encodings
    consulta = ler_arquivo_com_fallback(consulta_path).strip()

    # Processar consulta e salvar resultado em resposta.txt
    resultado = processar_consulta(consulta, indice)

    if resultado is None:
        resultado = set()

    with open("resposta.txt", "w") as f:
        f.write(f"{len(resultado)}\n")
        for doc_id in sorted(resultado):
            f.write(f"{doc_paths[doc_id-1]}\n")

if __name__ == "__main__":
    main()
