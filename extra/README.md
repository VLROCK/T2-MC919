# Relatório técnico em LaTeX

O arquivo `relatorio.tex` é um rascunho da parte técnica da Variante A. Ele contém a metodologia, as fórmulas, o plano experimental e a discussão sobre a comparação com scanners laser terrestres. Os campos vermelhos `[PREENCHER: ...]` dependem da captura e dos resultados reais; os campos azuis `[OPCIONAL: ...]` indicam extensões que só devem ser mantidas se forem executadas.

Antes da entrega:

1. Preencher integrantes, ambiente, câmera, número de fotos, parâmetros e todos os resultados medidos.
2. Inserir figuras de keypoints, matches, nuvem, trajetória e planta baixa. Remover hipóteses ou extensões não executadas.
3. Ajustar as tabelas ao estudo de ablação efetivamente realizado.
4. Compilar duas vezes com `pdflatex -interaction=nonstopmode relatorio.tex`, revisar o PDF e confirmar o limite de **6 páginas**.
5. Conferir se cada valor em centímetros está em escala métrica e se a dimensão usada para calibrar essa escala não foi reaproveitada como evidência de acurácia.

Neste ambiente, a compilação local não pôde ser validada: o MiKTeX falha na inicialização por acesso negado ao registro e informa que a instalação ainda precisa ser configurada. O arquivo `.tex` foi conferido estruturalmente, mas a paginação final depende de uma compilação funcional.

## Comparação opcional com scanner laser terrestre (TLS)

Se houver uma nuvem TLS **da mesma sala e da mesma disposição dos objetos**, exporte as duas nuvens em PLY com coordenadas em metros. Copie `tls_landmarks.example.json` para um novo arquivo e substitua os três pontos fictícios por coordenadas de alvos físicos correspondentes, visíveis em ambas as nuvens. Escolha marcos não colineares e bem distribuídos pela cena.

```bash
python -m src.compare_tls \
  --photogrammetry work/main/analysis/scaled_sparse.ply \
  --tls data/tls/scan.ply \
  --landmarks data/tls/landmarks.json \
  --output work/main/tls_comparison.json \
  --aligned-ply work/main/photo_in_tls.ply
```

A rotina lê PLY ASCII ou binário com vértices primeiro e propriedades escalares, faz amostragem reproduzível e redução por voxels, alinha por rotação/translação **sem alterar a escala**, e reporta resíduos dos marcos e distâncias de vizinho mais próximo em ambas as direções. A nuvem fotogramétrica esparsa tem menor cobertura que um TLS: principalmente a distância TLS→fotogrametria indica lacunas de amostragem, não um erro de precisão isolado. Inspecione a sobreposição visualmente e compare também medidas físicas independentes.
