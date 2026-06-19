Este repositório contém um projeto de visão computacional aplicada à engenharia civil, com foco na classificação automática de patologias em estruturas de concreto a partir de imagens reais coletadas em campo.

A ideia aqui é simples:
👉 usar inteligência artificial para ajudar na inspeção estrutural, reduzindo a dependência exclusiva da avaliação visual humana, que muitas vezes é subjetiva e demorada.

As patologias consideradas neste trabalho são:

Fissuras
Corrosão de armaduras
Desplacamento do concreto
🎯 Motivação

A inspeção de estruturas de concreto ainda é, em grande parte, baseada em observações visuais feitas por especialistas. Apesar de eficaz, esse processo pode variar bastante de acordo com a experiência do inspetor.

Com os avanços recentes em aprendizado profundo e visão computacional, tornou-se viável desenvolver modelos capazes de analisar imagens e identificar padrões associados a danos estruturais.

Este projeto explora exatamente isso:
até que ponto um modelo de deep learning consegue aprender a diferenciar patologias comuns em estruturas de concreto?

🧠 O que foi feito
Organização e pré-processamento das imagens com OpenCV
Análise exploratória e geração de gráficos com Matplotlib
Treinamento supervisionado de uma Rede Neural Convolucional
Utilização da arquitetura ResNet-18, amplamente validada na literatura
Avaliação do desempenho com métricas clássicas de classificação
Análise visual de exemplos corretos e erros do modelo

O treinamento foi realizado com Early Stopping, interrompendo automaticamente quando o desempenho em validação deixou de evoluir.

📊 Resultados (resumo rápido)

O modelo apresentou um desempenho consistente no conjunto de validação:

Acurácia geral: 93,3%
Excelente desempenho na classe fissura
Pequena confusão entre corrosão e desplacamento, o que é esperado devido à similaridade visual em alguns casos reais

Esses resultados indicam que o modelo é promissor como ferramenta de apoio à inspeção, embora não substitua a avaliação técnica especializada.

🗂️ Estrutura do Projeto
classificacao-patologias-concreto/
│
├── adaptar_dataset.py          # Organização e preparação do dataset
├── avaliacao_real.py           # Avaliação do modelo em imagens reais
├── pipeline.py                 # Pipeline inicial
├── pipeline_v2.py              # Pipeline final de treinamento
├── requirements.txt            # Dependências
│
├── dataset/                    # Dataset processado
├── imagens_brutas/             # Imagens originais por classe
│
├── modelos/
│   └── melhor_modelo.pth       # Modelo treinado (ResNet-18)
│
├── resultados/
│   ├── historico_treinamento.png
│   ├── matriz_confusao.png
│   └── exemplos_predicoes.png
│
└── docs/
    └── Resultados_Preliminares_Final.docx
🛠️ Tecnologias utilizadas
Python
PyTorch
OpenCV (processamento de imagens)
Matplotlib (visualização e gráficos)
NumPy
Scikit-learn
▶️ Como rodar o projeto

Clone o repositório:

git clone https://github.com/Marlonb87/classificacao-patologias-concreto.git

Instale as dependências:

pip install -r requirements.txt

Execute o pipeline principal:

python pipeline_v2.py
🚀 Próximos passos

Este trabalho é preliminar, e existem várias evoluções possíveis:

Migração do modelo para YOLO, permitindo não só classificar, mas localizar as patologias na imagem
Ampliação do dataset com mais imagens e maior diversidade de cenários
Aplicação futura em inspeções automatizadas (drones, dispositivos móveis, etc.)
👤 Autor

Marlon Barcelos
Engenharia • Visão Computacional • Inspeção Estrutural
