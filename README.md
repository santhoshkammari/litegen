# AiLite 🚀

AiLite is a unified interface for accessing state-of-the-art language models through popular AI frameworks. It provides seamless integration with frameworks like DSPy, LangChain, AutoGen, and LlamaIndex while making advanced AI models accessible and free to use.

[![PyPI version](https://badge.fury.io/py/ailite.svg)](https://badge.fury.io/py/ailite)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

## 🌟 Features

- **Universal Framework Support**: Compatible with major AI frameworks including DSPy, LangChain, AutoGen, and LlamaIndex
- **Access to Leading Models**: Support for 30+ cutting-edge language models from providers like:
  - Qwen (72B, 32B models)
  - Meta's Llama family
  - Google's Gemma series
  - Mistral and Mixtral
  - Microsoft's Phi models
  - And many more!
- **Framework-Native Integration**: Use models with your favorite framework's native interfaces
- **Consistent API**: Uniform experience across different frameworks
- **Free Access**: Leverage powerful AI models without cost barriers

## 📦 Installation

```bash
pip install ailite
```

## 🚀 Quick Start

### DSPy Integration

```python
from ailite.dspy import HFLM

model = HFLM(model="Qwen/Qwen2.5-72B-Instruct")
```

### LangChain Integration

```python
from ailite.langchain import ChatOpenAI

chat_model = ChatOpenAI(model="mistralai/Mixtral-8x7B-Instruct-v0.1")
```

### AutoGen Integration

```python
from ailite.autogen import OpenAIChatCompletionClient

client = OpenAIChatCompletionClient(model="meta-llama/Llama-3.1-70B-Instruct")
```

### LlamaIndex Integration

```python
from ailite.llamaindex import OpenAI

llm = OpenAI(model="google/gemma-2-9b-it")
```

## 🎯 Supported Models

AiLite supports a wide range of cutting-edge language models, including:

### Large Language Models
- Qwen (72B, 32B variants)
- Meta Llama 3 family
- Google Gemma series
- Mistral and Mixtral
- Microsoft Phi
- Yi models
- CodeLlama
- Falcon
- And many more!

For a complete list of supported models, check our [models documentation](docs/MODELS.md).

## 🛠️ Framework Support

Currently supported frameworks:
- DSPy
- LangChain
- AutoGen
- LlamaIndex

More frameworks coming soon!

## 📚 Documentation

For detailed documentation and examples, visit our [documentation site](docs/README.md).

### Examples

- [Basic Usage Examples](examples/basic_usage.md)
- [Framework-Specific Examples](examples/frameworks.md)
- [Advanced Usage Patterns](examples/advanced.md)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

Special thanks to the communities behind:
- DSPy
- LangChain
- AutoGen
- LlamaIndex

And to all the model providers for making their models accessible.

## 📫 Contact

- GitHub Issues: [Create an issue](https://github.com/yourusername/ailite/issues)
- Email: your.email@example.com

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/ailite&type=Date)](https://star-history.com/#yourusername/ailite&Date)

---

Made with ❤️