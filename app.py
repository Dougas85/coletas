import os
import re
from io import BytesIO
from flask import Flask, render_template, request, redirect, url_for, flash, send_file
import pandas as pd
from fpdf import FPDF
from unidecode import unidecode

# -----------------------------
# Configurações
# -----------------------------
DATA_DIR = "data"
BASE_TXT = os.path.join(DATA_DIR, "base.txt")   # seu TXT base (coloque esse arquivo)
BASE_CSV = os.path.join(DATA_DIR, "base.csv")   # será criado automaticamente
ALLOWED_EXT = {"txt", "csv"}

PDF_FILENAME = "repetidos_remetente.pdf"

app = Flask(__name__)
app.secret_key = "troque_esta_chave_para_producao"

# Variáveis globais em memória (apenas para uso local/testes)
DF_BASE = None      # DataFrame da base histórica
DF_MATCH = None     # Resultado do cruzamento (linhas repetidas do arquivo do dia)

# -----------------------------
# Utilitários
# -----------------------------
def try_decode_bytes(b: bytes, encodings=('utf-8', 'latin-1', 'cp1252')):
    for enc in encodings:
        try:
            return b.decode(enc)
        except Exception:
            continue
    # último recurso
    return b.decode('utf-8', errors='ignore')

def encontrar_linha_cabecalho(linhas):
    """
    Procura a linha que contém os nomes das colunas: 'Remetente' e 'Endereço' e 'CEP'
    Retorna índice da linha do cabeçalho ou 0 se não achar.
    """
    for i, l in enumerate(linhas):
        low = l.lower()
        if 'remetente' in low and ('endereço' in low or 'endereco' in low) and 'cep' in low:
            return i
    return 0

def split_linha(linha):
    """
    Tenta separar uma linha por tab. Se não tiver, separa por 2+ espaços.
    Retorna lista de colunas.
    """
    if '\t' in linha:
        return re.split(r'\t+', linha)
    # se o texto tiver campo com vírgulas e espaços, usar separador por 2+ espaços
    cols = re.split(r'\s{2,}', linha)
    # se ainda resultar em 1 coluna, tenta separar por tabulações invisíveis
    if len(cols) == 1:
        return re.split(r'\s{1,}', linha)
    return cols

def parse_txt_to_df(path_or_bytes, is_bytes=False):
    """
    Converte arquivo TXT (caminho ou bytes) em DataFrame padronizado.
    Retorna DataFrame com pelo menos as colunas:
      - Remetente
      - Endereço Origem (EnderecoOrigem)
      - CEP Origem (CEPOrigem)
    Também cria colunas normalizadas: remetente_norm, endereco_norm, cep_norm, chave
    """
    if is_bytes:
        text = try_decode_bytes(path_or_bytes)
        linhas = [l.rstrip('\n\r') for l in text.splitlines() if l.strip() != ""]
    else:
        with open(path_or_bytes, 'rb') as f:
            raw = f.read()
        text = try_decode_bytes(raw)
        linhas = [l.rstrip('\n\r') for l in text.splitlines() if l.strip() != ""]

    if not linhas:
        return pd.DataFrame()

    idx = encontrar_linha_cabecalho(linhas)
    header = split_linha(linhas[idx])
    data_lines = linhas[idx+1:]
    dados = []
    for l in data_lines:
        cols = split_linha(l)
        # padroniza tamanho
        if len(cols) < len(header):
            cols += [''] * (len(header) - len(cols))
        dados.append(cols[:len(header)])

    df = pd.DataFrame(dados, columns=header)

    # Normalizar nomes de colunas relevantes (tolerante)
    colmap = {}
    for c in df.columns:
        c_low = c.strip().lower()
        if 'remetent' in c_low:
            colmap[c] = 'Remetente'
        elif 'endereço orig' in c_low or 'endereco orig' in c_low or ('endereço' in c_low and 'orig' in c_low):
            colmap[c] = 'EnderecoOrigem'
        elif 'cep orig' in c_low or ('cep' in c_low and 'orig' in c_low):
            colmap[c] = 'CEPOrigem'
        elif 'endereço' in c_low and 'dest' in c_low:
            colmap[c] = 'EnderecoDestino'
        elif 'destinat' in c_low:
            colmap[c] = 'Destinatario'
        else:
            # mantém o nome original
            colmap[c] = c

    df = df.rename(columns=colmap)

    # Garantir colunas necessárias
    for need in ('Remetente', 'EnderecoOrigem', 'CEPOrigem'):
        if need not in df.columns:
            df[need] = ''

    # limpeza básica
    def clean_txt(s):
        if pd.isna(s):
            return ''
        s2 = str(s).strip()
        # remover underscores que aparecem nos nomes
        s2 = s2.replace('_', ' ')
        # normalizar espaços
        s2 = re.sub(r'\s+', ' ', s2)
        return s2

    df['Remetente'] = df['Remetente'].apply(clean_txt)
    df['EnderecoOrigem'] = df['EnderecoOrigem'].apply(clean_txt)
    df['CEPOrigem'] = df['CEPOrigem'].apply(clean_txt)

    # normalizações para comparação (remove acentos, upper, remove punctuation except digits)
    def norm_text(s):
        s = unidecode(s).upper()
        s = s.strip()
        s = re.sub(r'[^\w\s]', ' ', s)  # substitui pontuação por espaço
        s = re.sub(r'\s+', ' ', s)
        return s

    def norm_cep(s):
        if s is None:
            return ''
        s = re.sub(r'\D', '', str(s))
        return s.zfill(8) if s else ''

    df['remetente_norm'] = df['Remetente'].apply(norm_text)
    df['endereco_norm'] = df['EnderecoOrigem'].apply(norm_text)
    df['cep_norm'] = df['CEPOrigem'].apply(norm_cep)

    df['chave'] = df['remetente_norm'] + '|' + df['endereco_norm'] + '|' + df['cep_norm']

    return df

def ensure_base_loaded():
    """
    Garante que DF_BASE esteja carregado. Se existir base.csv, carrega.
    Se não existir, tenta converter base.txt -> base.csv automaticamente.
    """
    global DF_BASE
    if DF_BASE is not None:
        return

    if not os.path.isdir(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(BASE_CSV):
        try:
            DF_BASE = pd.read_csv(BASE_CSV, dtype=str).fillna('')
            # recalcular chave se necessário
            if 'chave' not in DF_BASE.columns:
                DF_BASE = parse_txt_to_df(BASE_CSV, is_bytes=False)
        except Exception:
            # se falhar em ler CSV, reconverter a partir do TXT
            DF_BASE = None

    if DF_BASE is None:
        if not os.path.exists(BASE_TXT):
            print("Arquivo base.txt não encontrado em 'data/base.txt'. Coloque-o lá e reinicie.")
            DF_BASE = pd.DataFrame(columns=['Remetente','EnderecoOrigem','CEPOrigem','chave'])
            return
        # parsear TXT e salvar CSV
        print("Convertendo base TXT -> CSV (uma vez). Aguarde...")
        dfb = parse_txt_to_df(BASE_TXT, is_bytes=False)
        # salvar CSV limpo
        try:
            dfb.to_csv(BASE_CSV, index=False)
            print(f"Base convertida e salva em {BASE_CSV}")
        except Exception as e:
            print("Erro ao salvar CSV da base:", e)
        DF_BASE = dfb

# -----------------------------
# Rotas Flask
# -----------------------------
@app.route('/', methods=['GET'])
def index():
    ensure_base_loaded()
    base_count = len(DF_BASE) if DF_BASE is not None else 0
    return render_template('index.html', base_count=base_count)

@app.route('/upload_dia', methods=['POST'])
def upload_dia():
    global DF_MATCH
    ensure_base_loaded()
    if 'file' not in request.files:
        flash("Nenhum arquivo enviado.", "danger")
        return redirect(url_for('index'))
    f = request.files['file']
    if f.filename == '':
        flash("Nenhum arquivo selecionado.", "warning")
        return redirect(url_for('index'))

    ext = f.filename.rsplit('.', 1)[-1].lower() if '.' in f.filename else ''
    if ext not in ALLOWED_EXT:
        flash("Formato não permitido. Use .txt ou .csv", "danger")
        return redirect(url_for('index'))

    try:
        raw = f.read()
        df_dia = parse_txt_to_df(raw, is_bytes=True)
    except Exception as e:
        flash(f"Erro ao ler arquivo do dia: {e}", "danger")
        return redirect(url_for('index'))

    # criar conjunto de chaves da base (para velocidade)
    base_keys = set(DF_BASE['chave'].astype(str).tolist())

    # marcar repetidos
    df_dia['is_repetido'] = df_dia['chave'].apply(lambda k: k in base_keys)

    df_repetidos = df_dia[df_dia['is_repetido']].copy().reset_index(drop=True)

    DF_MATCH = df_repetidos  # guarda no global para gerar PDF / download
    count = len(df_repetidos)
    flash(f"Processado. {count} endereços já constam na base histórica.", "success")
    # mostra a página de resultado com amostra
    html_table = df_repetidos[['Remetente','EnderecoOrigem','CEPOrigem']].head(500).to_html(classes='table table-sm table-striped', index=False, escape=False)
    return render_template('resultado.html', table=html_table, total=count)

@app.route('/download_pdf', methods=['GET'])
def download_pdf():
    global DF_MATCH
    if DF_MATCH is None or len(DF_MATCH) == 0:
        flash("Nenhum registro repetido disponível. Faça o upload do arquivo do dia e execute o cruzamento.", "warning")
        return redirect(url_for('index'))

    # Gerar PDF com os repetidos (Remetente — EnderecoOrigem — CEPOrigem)
    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.set_auto_page_break(auto=True, margin=10)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, "Relatório de Endereços Repetidos (Remetente — Endereço Origem — CEP Origem)", ln=True)
    pdf.ln(2)
    pdf.set_font("Arial", size=9)

    # colunas que vamos usar
    rows = DF_MATCH[['Remetente','EnderecoOrigem','CEPOrigem']].fillna('').values.tolist()

    # imprimir linhas com multi_cell
    for i, r in enumerate(rows):
        remetente = str(r[0])[:200]
        endereco = str(r[1])[:300]
        cep = str(r[2])[:12]
        linha = f"{remetente} — {endereco} — {cep}"
        pdf.multi_cell(0, 5, linha)
        # opcional: limite por página já tratado pelo FPDF

    # resumo
    pdf.ln(4)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 6, f"Total de registros repetidos: {len(rows)}", ln=True)

    bio = BytesIO()
    pdf.output(bio)
    bio.seek(0)
    return send_file(bio, as_attachment=True, download_name=PDF_FILENAME, mimetype='application/pdf')

# -----------------------------
# Execução
# -----------------------------
if __name__ == '__main__':
    ensure_base_loaded()
    app.run(debug=True, port=5000)

