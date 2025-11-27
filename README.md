# AI Magic Mask

AI Magic Mask é um app de vídeo em tempo real (GUI + CLI) que captura qualquer câmera (USB, laptop, IP/RTSP), detecta rostos e aplica efeitos e anonimização com alta performance.

Destaques
- Captura de múltiplas fontes: webcam (0, 1…), arquivo de vídeo, RTSP/IP.
- Detecção facial moderna:
  - MediaPipe Face Mesh (opcional, alta precisão)
  - Fallback leve com OpenCV Haar (frontal + perfil, cobre mais ângulos)
- Overlays em tempo real: caixas, tint, landmarks, HUD de FPS.
- Anonimização facial multi-rosto: blur forte, pixelado, sólido (cinza).
- Modo sem Tkinter (--cv2) para máxima compatibilidade no macOS.
- Foco em 30 FPS (ajuste resolução e backend para o seu hardware).

Requisitos (macOS recomendação)
- Python 3.9+ (recomendado instalar pelo site python.org – build universal2)
- Dependências:
  - opencv-python
  - numpy
  - Pillow
  - (Opcional) mediapipe

Instalação

```zsh
# 1) Crie e ative um ambiente virtual
python3 -m venv .venv
source .venv/bin/activate

# 2) Atualize pip e instale dependências essenciais
pip install --upgrade pip
pip install -r requirements.txt

# 3) (Opcional) Instale MediaPipe (se falhar, pule e use Haar)
# pip install mediapipe
```

Executando

GUI (Tkinter)
- Em macOS, a GUI funciona melhor com pythonw quando o Tk do sistema está incompatível.

```zsh
source .venv/bin/activate
pythonw magic_mask.py
# ou:
python3 magic_mask.py
```

OpenCV (sem Tkinter)
- Evita problemas de Tk/Tcl no macOS.

```zsh
source .venv/bin/activate
python3 magic_mask.py --cv2 --source 0 --res 720p --backend haar
```

Parâmetros úteis (CLI)
- --source: 0|1|/path/video.mp4|rtsp://user:pass@host:port/stream
- --res: 480p|720p
- --backend: haar|mediapipe
- --anonymize: off|blur|pixel|solid
- --max-frames: N (fecha automático após N frames; útil para testes não-interativos)

Anonimização (não parecer rosto)
- blur: borrado forte, duas passagens de GaussianBlur com kernel grande adaptativo.
- pixel: pixelado forte (rápido e eficiente).
- solid: cobertura sólida cinza (muito leve).
- Múltiplos rostos são processados, com máscara via hull dos landmarks (MediaPipe) ou elipse sobre bbox (Haar).

Exemplos

```zsh
# Webcam com blur forte (Haar)
python3 magic_mask.py --cv2 --source 0 --anonymize blur

# Pixelar (mais leve)
python3 magic_mask.py --cv2 --source 0 --anonymize pixel

# Sólido cinza
python3 magic_mask.py --cv2 --source 0 --anonymize solid

# MediaPipe (se instalado), melhor máscara em todos os ângulos
python3 magic_mask.py --cv2 --source 0 --backend mediapipe --anonymize blur
```

Controles
- Modo OpenCV: q sai; s salva screenshot.
- GUI: botões Iniciar/Parar/Screenshot; seletores de fonte, resolução, backend e anonimização.

Desempenho
- 480p + Haar + pixel/solid: mais chance de 30 FPS.
- 720p + Haar + blur: 20–35 FPS (depende de CPU); blur é mais pesado.
- MediaPipe é mais preciso, porém mais pesado; prefira 480p se necessário.

Compatibilidade macOS
- Erro comum: `macOS 26 (2601) or later required, have instead 16 (1601)` ao usar Tk.
  - Soluções:
    - Instale Python oficial (python.org) e rode com `pythonw`.
    - Use `--cv2` para evitar Tk.
- Permissões de câmera: System Settings → Privacy & Security → Camera → permitir Python do seu venv.

Problemas comuns
- "Erro ao abrir fonte de vídeo": verifique se outra app não está usando a câmera; confira permissões; para RTSP, a rede e build do OpenCV com FFmpeg importam.
- Falha ao instalar MediaPipe (jax/jaxlib): mantenha Haar; tente com Python 3.10+ depois.

Arquitetura rápida
- magic_mask.py: app completo (GUI + CLI), detecção, overlays e anonimização.
- Backends:
  - MediaPipe Face Mesh (convex hull dos landmarks)
  - Haar Cascades (frontal + perfil via flip; NMS simples para fundir detecções)

Roadmap (opcional)
- Hotkeys para alternar modos de anonimização em tempo real no --cv2.
- Smoothing de landmarks (EMA/Kalman) para bordas estáveis.
- Efeitos 2D/3D com warping (óculos, máscaras PNG) e partículas.
- Aceleração com shaders (OpenGL/WebGPU) ou export para Unity/Unreal.

Licença
- MIT

