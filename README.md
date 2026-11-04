# Reconstrução 3D de cena forense simulada (MC919 T2, Variante A)

## Dados capturados em 3 de novembro

As duas pastas em `data/T2_MC919-*` são uma única sequência do iPhone 17 Pro.
Há 1.161 JPGs extraídos e um MOV; os dois ZIPs contêm as mesmas fotos já
extraídas. `src/prepare_images.py` ordena pelos números `IMG_####`, registra
origens e seleciona até 400 vistas em `data/images`, preservando todas as fotos
nas quais detectou o ArUco. O CSV `data/image_preparation.csv` permite revisar
a seleção e uma medida indicativa de nitidez. As medidas de campo informadas
foram 4,00 × 4,30 × 2,75 m e estão em `data/measurements.json`.

O grupo mediu 7 × 7 cm da **área branca inteira** do marcador exibido na tela.
O SVG tem 190 mm de lado total e 160 mm de lado preto; se a área medida for
exatamente a borda do SVG sem margens adicionais do visualizador, o lado preto
seria `0,07 × 160/190 = 0,05895 m`. Confirmem diretamente o lado preto na
tela antes de usar o valor como escala final. É possível refazer apenas a
escala e a análise depois, sem repetir o COLMAP.

### Execução recomendada no cluster

O GitHub contém código e manifestos; copiem manualmente as 400 fotos de
`data/images/` para a mesma pasta no cluster antes de enviar os jobs. O caminho
de Python abaixo usa o ambiente mostrado no exemplo do grupo:

No computador local, um exemplo de cópia é
`scp -r data/images victor.roquetto@HOST_DO_CLUSTER:/home/victor.roquetto/T2-MC919/data/`.
Substituam `HOST_DO_CLUSTER` pelo endereço real usado por vocês.

```bash
git clone https://github.com/VLROCK/T2-MC919.git
cd T2-MC919
mkdir -p data/images
# Copie os 400 JPGs locais para esta pasta antes de continuar.
test "$(find data/images -maxdepth 1 -iname '*.JPG' | wc -l)" -eq 400
export PY="$HOME/miniconda3/envs/horus_theme/bin/python"
"$PY" -m pip install -r requirements.txt
export COLMAP_BIN="$HOME/miniconda3/envs/t2_colmap/bin/colmap"
"$COLMAP_BIN" help >/dev/null
mkdir -p results/logs
bash scripts/submit_cluster.sh --gres=gpu:1
```

O primeiro job usa matching exaustivo nas 400 fotos e gera nuvem esparsa,
poses, métricas e vídeo. O segundo depende do sucesso do primeiro e executa
MVS e malha. A textura fotográfica requer um COLMAP com `mesh_texturer` (4.0+);
o executável 3.13 usado no cluster não oferece esse comando. `--gres=gpu:1` é
um exemplo: ajustem essa opção à
partição/GPU exigida pelo cluster. Sem GPU, retirem a opção e esperem uma
execução mais lenta; MVS pode não estar disponível na instalação do COLMAP.
Os recursos padrão nos scripts são 16 CPUs e 64 GB; também podem ser ajustados.
Após um OOM na extração em CPU, o job esparso limita a imagem de extração a
2000 px, SIFT a 4096 características por foto e a extração a quatro threads.
Os parâmetros ficam em `manifest.json` e podem ser ajustados por
`FEATURE_IMAGE_SIZE`, `MAX_FEATURES`, `FEATURE_THREADS` e `MAX_MATCHES`.
Os IDs impressos pelo script permitem acompanhar com `squeue -j ID` e os logs
ficam em `results/logs/`.
`submit_cluster.sh` agora verifica o executável COLMAP e as 400 fotos **antes**
de chamar `sbatch`.

Se o COLMAP já concluiu a reconstrução esparsa em `work/gpu_compat` e o job
falhou somente na análise posterior, aproveitem o modelo existente. No cluster:

```bash
cd /home/victor.roquetto/T2-MC919
git pull origin main
export PY="$HOME/miniconda3/envs/horus_theme/bin/python"
export COLMAP_BIN="$HOME/miniconda3/envs/t2_colmap/bin/colmap"
WORK_DIR=work/gpu_compat bash scripts/submit_resume.sh --gres=gpu:1
```

`FINISH_JOB` calcula métricas, vídeo e comparação SIFT/ORB/AKAZE em CPU;
`DENSE_JOB` inicia MVS e malha em GPU após o primeiro terminar; só produz textura
se o COLMAP instalado oferecer `mesh_texturer`. Nenhum
dos dois refaz extração de características, matching ou mapeamento esparso.
Consultem `squeue -j ID` e `results/logs/t2-finish-ID.out` ou
`results/logs/t2-mvs-ID.out`. A análise de descritores passa a respeitar a
orientação de pixels usada pela calibração do COLMAP, mesmo quando o EXIF pede
rotação para visualização.

Se um job denso anterior falhou apenas em `mesh_texturer` com o COLMAP 3.13,
`fused.ply` e `meshed-delaunay.ply` já estão prontos. Após atualizar o código,
finalizem os metadados e as métricas em um job curto, sem GPU nem repetição do MVS:

```bash
cd /home/victor.roquetto/T2-MC919
git pull origin main
export PY="$HOME/miniconda3/envs/horus_theme/bin/python"
export COLMAP_BIN="$HOME/miniconda3/envs/t2_colmap/bin/colmap"
export WORK_DIR=work/gpu_compat
mkdir -p results/logs
sbatch --parsable --export=ALL --output=results/logs/t2-finalize-%j.out \
  scripts/slurm_finalize_dense.sh
```

O resumo passará a registrar `dense_ply`, `mesh_ply` e
`texture_status: unavailable_in_colmap`. O job antigo continuará marcado como
falho no `sacct`; o novo job registra a finalização.
Se uma versão 4.0+ do COLMAP estiver disponível em outro ambiente, somente a
etapa `mesh_texturer` precisa ser executada sobre `dense/`; a nuvem e a malha
existentes podem ser reaproveitadas. Confiram primeiro `COLMAP4 mesh_texturer -h`
e mantenham a pasta `dense/images` com as imagens retificadas.
Se houver apenas uma compilação CPU do COLMAP, usem
`CPU_ONLY=1 SKIP_DENSE=1 bash scripts/submit_cluster.sh` para executar o
modelo esparso; a densificação do COLMAP requer CUDA.

O job aceita `export MARKER_SIZE_M=...` antes do `sbatch`; deixem sem valor
até confirmar o lado preto. A escala pode ser aplicada depois sem refazer SfM.

Depois de conferir o **lado preto** do marcador, calculem a escala e refaçam
as medidas (substituam o número de exemplo):

```bash
python -m src.scale_marker --model work/main/sparse_txt --images data/images \
  --marker-id 0 --marker-size-m 0.05895 --output work/main/scale.json
python -m src.analysis --model work/main/sparse_txt --manifest work/main/manifest.json \
  --measurements data/measurements.json --scale-json work/main/scale.json \
  --output work/main/analysis
python -m src.visualize --model work/main/sparse_txt \
  --metrics work/main/analysis/metrics.json \
  --output work/main/analysis/navigation.mp4
```

Para cores mais contínuas e uma malha, rodem a etapa pesada depois:

```bash
python -m src.densify --work work/main --mesh
```

Ela produz `dense/fused.ply` (pontos densos coloridos) e
`dense/meshed-delaunay.ply` (superfície). Com COLMAP 4.0+, também produz
`dense/textured/` (malha e textura fotográfica).
Áreas sem fotos úteis poderão continuar com buracos ou artefatos. Após MVS,
repitam `src.analysis` para atualizar a contagem de pontos densos.

Projeto para reconstruir a sala encenada com SfM, obter escala métrica, medir
dimensões e avaliar a qualidade. O arquivo `T2_2026.pdf` é o enunciado. As
fotos selecionadas e as medidas da sala estão neste repositório; o tamanho
físico do ArUco ainda precisa de confirmação.

## O que é entregue

| Componente | Implementação | Saída |
|---|---|---|
| SIFT/ORB/AKAZE, matching, F/E, poses e triangulação de pares | `src/feature_analysis.py` | CSV, JSON, imagens de keypoints/inliers/epipolares e PLY de dois quadros |
| SfM COLMAP, matching sequencial/exaustivo e MVS opcional | `src/pipeline.py` | modelo esparso, poses, banco e nuvem densa opcional |
| Escala automática com ArUco | `src/scale_marker.py` | fator em m/unidade SfM e diagnósticos |
| Métricas, planos RANSAC, planta, distâncias entre pontos de interesse | `src/analysis.py` | JSON, PNG e PLY métricos |
| Navegação virtual por poses estimadas | `src/visualize.py` | MP4 de 30 a 60 s |
| Relatório técnico | `extra/relatorio.tex` | texto LaTeX com campos para resultados reais |

A visualização é da **nuvem esparsa**, não uma malha fotorealista. O vídeo percorre
poses interpoladas; partes sem pontos aparecerão vazias. A nuvem densa é
recomendada no enunciado e pode melhorar a cobertura visual, mas não é exigida
para a variante A.

## Preparação

1. Instale Python 3.10+, as dependências e **COLMAP**. Confira `colmap help`
   (`COLMAP.bat help` no Windows, se for o executável instalado). As opções de
   COLMAP variam entre versões; os comandos usados ficam em `commands.log`.

   ```bash
   python -m pip install -r requirements.txt
   colmap help
   ```

2. Registre a sala com o professor. Prefira uma sala com móveis e textura.
   Monte a cena simulada e mantenha os objetos imóveis. Evite pessoas andando,
   vidro, espelhos, superfícies brilhantes e contraluz.
3. Gere um marcador físico, imprima em **100%**, sem ajuste à página, cole em
   suporte rígido e **meça com régua o lado preto que saiu da impressora**:

   ```bash
   python -m src.make_marker --id 0 --side-mm 160 --output data/marker_0.svg
   ```

   O SVG gerado mede 190 mm com a margem branca e cabe em A4 sem redução em
   impressoras com margem útil suficiente. O valor passado ao estimador deve ser o *lado impresso medido*, por exemplo
   `0.158` m, e não necessariamente os 160 mm nominais do SVG. Deixe o marcador
   visível em pelo menos três fotos registradas, com posições de câmera
   suficientemente diferentes. Um marcador grande reduz a incerteza em uma
   captura de sala.
4. Coloque 100–300 imagens em `data/images/` com nomes em ordem temporal, como
   `frame_0001.jpg`. Caminhe em laço fechado com 60–80% de sobreposição, câmera
   voltada a paredes e móveis, mantendo lente/zoom e iluminação. **Desloque-se**
   entre fotos. Fotografe início e fim próximos e com elementos comuns. Meça
   largura, comprimento e uma altura com trena; guarde anotações independentes
   do marcador de escala.

## Execução principal

Em Bash/Git Bash/WSL, a partir da raiz do projeto:

```bash
MARKER_SIZE_M=0.158 MARKER_ID=0 MATCHING=sequential bash run.sh
```

No PowerShell, o comando equivalente é:

```powershell
.\run.ps1 -MarkerSizeM 0.158 -MarkerId 0 -Matching sequential
```

Sem GPU compatível, acrescente `CPU_ONLY=1`. Para MVS, acrescente `DENSE=1`;
isso aumenta bastante o processamento. No Windows, defina `COLMAP_BIN` com o
caminho do `COLMAP.bat` se `colmap` não estiver no PATH. Também é possível
executar cada módulo com `python -m src.<modulo> --help` no PowerShell.

`run.sh` chama COLMAP, detecta escala ArUco quando `MARKER_SIZE_M` foi
informado, compara descritores, gera métricas/plantas e produz o vídeo. Não
reutilize o mesmo `WORK_DIR` entre experimentos: o pipeline recusa uma pasta
ocupada para não misturar resultados.

Se não usarem ArUco, copiem `data/measurements.example.json` para
`data/measurements.json` e substituam os IDs de `reference.point_ids` por dois
pontos identificáveis em `work/main/sparse_txt/points3D.txt`, com a distância
real entre eles. A escala fica `L_real / distância_SfM`. Os IDs de `dimensions`
e `evidence` são opcionais; retirem entradas ainda não anotadas. A seção
`room_ground_truth_m` guarda as medidas de trena para comparar com os planos
de paredes detectados automaticamente. Se o programa não detectar dois pares
de paredes confiáveis, `width_m`/`length_m` serão `null` e a planta deve ser
medida com pontos 3D escolhidos manualmente.

Arquivos principais:

```text
work/main/manifest.json            configurações, tempos e modelo escolhido
work/main/commands.log             comandos e mensagens COLMAP
work/main/sparse_txt/              modelo COLMAP legível
work/main/scale.json               fator métrico e qualidade do ArUco
work/main/features/                comparação dos descritores
work/main/analysis/metrics.json    métricas e medidas
work/main/analysis/floorplan.png   projeção superior
work/main/analysis/scene3d.png     pontos, poses e trajetória
work/main/analysis/coverage.png    densidade de pontos projetados
work/main/analysis/camera_poses.json  poses métricas completas
work/main/analysis/navigation.mp4  vídeo de navegação
```

## Estudos comparativos

**Descritores.** `feature_analysis.py` usa os mesmos pares para SIFT, ORB e
AKAZE e mede keypoints, matches após ratio test, inliers de F/E, triangulações
válidas e tempo. A calibração estimada pelo COLMAP é exportada para
`work/main/intrinsics.json` quando há uma única câmera compatível. Sem ela, a
pose de dois quadros é apenas exploratória. Esse estudo local **não troca o
descritor dentro do COLMAP**: o pipeline global usa o extrator configurado no
COLMAP.

**Matching e fotos.** Rode duas execuções em pastas diferentes, depois compare
imagens registradas, pontos, erro métrico independente, deriva e tempo:

```bash
WORK_DIR=work/seq MATCHING=sequential MARKER_SIZE_M=0.158 bash run.sh
WORK_DIR=work/exh MATCHING=exhaustive MARKER_SIZE_M=0.158 bash run.sh
python -m src.compare_runs --runs work/seq work/exh --output work/comparison
```

Para ablação por quantidade, use `MAX_IMAGES=100` e outro `WORK_DIR`. A amostra
é uniforme sobre a sequência. `LOOP_DETECTION=1` habilita detecção de laço no
matcher sequencial, mas pode precisar baixar uma vocabulary tree; a opção
normalmente vem desligada. O matching exaustivo cresce aproximadamente com
o número de pares `N(N-1)/2`.

**TLS profissional.** Se houver scan laser da *mesma* sala, use
`src/compare_tls.py` após estabelecer três ou mais alvos correspondentes. A
comparação deve ser feita nas regiões comuns e com controle de densidade;
sem scan real, a seção do relatório é qualitativa. Um scanner laser terrestre
mede distâncias ativamente e exige sua própria verificação de precisão. Não
atribua números de exatidão ao equipamento sem medição.

## Limites e interpretação

- SfM monocular tem escala indeterminada. ArUco automatiza a detecção da
  referência, mas o tamanho físico do marcador continua obrigatório.
- A planta automática ajusta planos por RANSAC na nuvem esparsa. Móveis,
  paredes incompletas e uma câmera muito inclinada podem causar pares falsos;
  revise as imagens, os planos em `metrics.json` e as dimensões antes do
  relatório. Não trate valor automático como verdade de campo.
- A deriva entre primeira e última pose inclui o pequeno deslocamento físico
  entre fotos e só faz sentido se ambas forem registradas.
- Um erro de reprojeção baixo não garante medidas corretas. Compare sempre
  largura, comprimento e altura contra trena **independente da referência de
  escala**.
- Marcadores de evidência pequenos podem aparecer nas fotos sem produzir
  pontos 3D estáveis. Marque no relatório quais localizações foram anotadas
  manualmente e quais foram inferidas.

O modelo serve para demonstrar reconstrução e medição de uma **cena simulada**.
Não reivindica validação como instrumento de perícia.
