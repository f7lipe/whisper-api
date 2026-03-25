# Whisper Transcription API

API de transcrição de áudio para **Português do Brasil** usando [OpenAI Whisper](https://github.com/openai/whisper), com interface web de teste integrada.

## Requisitos

- Python 3.11+
- [ffmpeg](https://ffmpeg.org/) (`brew install ffmpeg`)

## Setup

```bash
# 1. Criar e ativar ambiente virtual
python -m venv venv
source venv/bin/activate

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Configurar variáveis (opcional — defaults já funcionam)
cp .env.example .env

# 4. Iniciar servidor
uvicorn app.main:app --reload
```

O servidor sobe em **http://localhost:8000**.  
Na primeira inicialização, o modelo `turbo` (~800 MB) é baixado automaticamente.

## Endpoints

| Método | Rota | Descrição |
|--------|------|-----------|
| `GET` | `/api/health` | Status do servidor e modelo |
| `POST` | `/api/transcribe` | Transcrever arquivo de áudio |

### POST `/api/transcribe`

**Request:** `multipart/form-data` com campo `file` (qualquer formato de áudio: mp3, wav, webm, m4a, flac, ogg)

**Response:**
```json
{
  "text": "Transcrição completa aqui",
  "language": "pt",
  "duration": 1.234,
  "segments": [
    { "id": 0, "start": 0.0, "end": 3.5, "text": "Texto do segmento" }
  ]
}
```

### Testar via curl

```bash
curl -X POST http://localhost:8000/api/transcribe \
  -F "file=@/caminho/para/audio.mp3"
```

## Interface Web

Acesse **http://localhost:8000** para usar a interface de teste com:
- Gravação direta pelo microfone
- Upload de arquivo de áudio
- Visualizador de áudio em tempo real
- Resultado da transcrição na tela

## Configuração (`.env`)

| Variável | Default | Descrição |
|----------|---------|-----------|
| `WHISPER_MODEL` | `turbo` | Modelo Whisper (`tiny`, `base`, `small`, `medium`, `large`, `turbo`) |
| `WHISPER_LANGUAGE` | `pt` | Código do idioma |
| `HOST` | `0.0.0.0` | Host do servidor |
| `PORT` | `8000` | Porta do servidor |
