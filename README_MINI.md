# AI Magic Mask (Single-file)

Interface gráfica simples e estável para detecção facial com overlays em tempo real.

Requisitos:
- macOS com Python 3.9+ (recomendado Python oficial de python.org)
- pacotes: opencv-python, numpy, Pillow
- opcional: mediapipe (para Face Mesh com alta precisão)

Instalação rápida (recomendada no macOS):

```bash
# 1) Instale Python oficial (universal2) em https://www.python.org/downloads/mac-osx/
#    Ele inclui um Tkinter compatível. Evite o Python do Xcode/CommandLineTools.

# 2) Crie um ambiente virtual
python3 -m venv .venv
source .venv/bin/activate

# 3) Atualize pip e instale dependências essenciais
pip install --upgrade pip
pip install -r requirements.txt

# 4) (Opcional) Instale MediaPipe (pode baixar jax/jaxlib; se falhar, pule)
# pip install mediapipe
```

Executar (GUI Tkinter):

```bash
# Em alguns ambientes macOS, GUI com Tkinter funciona melhor via pythonw
# Use pythonw se houver erro de versão do macOS/Tk
source .venv/bin/activate
pythonw magic_mask.py

# Se pythonw não existir, tente
python3 magic_mask.py
```

Executar (sem Tkinter, janela OpenCV):

```bash
# Evita dependência do Tk/Tcl (útil se ocorrer o erro "macOS 26 ... required")
source .venv/bin/activate
python3 magic_mask.py --cv2 --source 0 --res 720p --backend haar

# Exemplos de fonte:
#   --source 0           (webcam)
#   --source /path/v.mp4 (arquivo)
#   --source rtsp://...  (IP cam)
# Resolução: --res 480p|720p
# Backend:   --backend haar|mediapipe
```

Anonimização (mascarar o rosto):
- Modos: off (desligado), blur (borrado), pixel (pixelado), solid (básico sólido cinza)
- GUI: selecione no combobox "Anonimizar"
- OpenCV: use `--anonymize {off|blur|pixel|solid}`

Exemplos (OpenCV):

```bash
# Borrar rostos
python3 magic_mask.py --cv2 --source 0 --anonymize blur

# Pixelar rostos
python3 magic_mask.py --cv2 --source 0 --anonymize pixel

# Cobrir com sólido
python3 magic_mask.py --cv2 --source 0 --anonymize solid
```

Uso (GUI):
- Campo "Fonte de Vídeo":
  - "0" → webcam padrão
  - "/path/video.mp4" → arquivo
  - "rtsp://..." → câmera IP
- Selecione a resolução (480p ou 720p)
- Selecione o backend de detecção (MediaPipe ou Haar)
  - Se MediaPipe não estiver instalado, Haar é usado automaticamente
- Se desejar anonimização, escolha o modo em "Anonimizar"
- Clique "Iniciar"
- "Screenshot" salva PNG na pasta atual

Atalhos (OpenCV):
- Tecla "q" → sair
- Tecla "s" → salvar screenshot

Notas de desempenho:
- Típico: 20–35 FPS em 720p (depende do hardware)
- MediaPipe é mais pesado; se o FPS ficar baixo, use 480p ou Haar
- Anonimizar adiciona custo: pixel/solid são mais leves, blur é mais pesado

Problemas comuns e soluções:
- Erro: `macOS 26 (2601) or later required, have instead 16 (1601)`
  - Causa: Tkinter/Tcl do sistema não compatível com sua versão de macOS.
  - Soluções:
    - Use Python oficial de python.org (universal2), que inclui Tk compatível.
    - Execute via `pythonw magic_mask.py` (melhor para apps GUI no macOS), ou use `--cv2` para evitar Tk.
    - Evite o Python do Xcode/CommandLineTools.
- "Erro ao abrir fonte de vídeo":
  - Verifique câmera livre e permissões (macOS: System Settings → Privacy & Security → Camera → permitir Python).
  - Para RTSP, a confiabilidade depende da rede e do build do OpenCV com FFmpeg.
- ImportError Pillow/Tkinter:
  - Use Python oficial (python.org) no macOS, que inclui Tk, ou use `--cv2`.
- MediaPipe falha para instalar (jax/jaxlib erros):
  - Mantenha o backend Haar; MediaPipe é opcional.
  - Alternativa: use Python 3.10+ ou tente instalar mediapipe depois.

Licença: MIT
