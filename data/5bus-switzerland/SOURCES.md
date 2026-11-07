# Fontes e preparação dos perfis suíços

Preparado em 30/10/2026 para treinamento por imitação com `single_phase_socp`.

## Tempo e unidades

`demand.csv` foi preservado. Seus timestamps são copiados literalmente para
`price.csv` e `pv.csv`, na mesma ordem: 70.176 intervalos de 15 minutos.
Cada timestamp identifica o início do intervalo. Em `Europe/Zurich`, o período
vai de 01/01/2023 00:00 até 01/01/2025 00:00, exclusivo. Em UTC, começa em
31/12/2022 23:00; isso pertence ao ano tarifário suíço de 2023.
O calendário inclui 29/02/2024 e as mudanças de horário de verão, sem duplicar
ou eliminar instantes UTC.

As colunas são `timestamp,price_per_kwh` e `timestamp,PV5`. O preço está em
CHF/kWh, sem IVA; `PV5` é a potência ativa AC disponível em kW, média do
intervalo, antes do despacho e de eventual corte. Os arquivos de entrada usam
unidades físicas; os módulos fazem a conversão para pu no contrato de execução.

## Preço para consumidores

Foi adotada a mediana nacional residencial H4 da ElCom, categoria de referência
com consumo de 4.500 kWh/ano:

| Ano no horário suíço | Referência (centavos CHF/kWh) | `price_per_kwh` (CHF/kWh) |
| --- | ---: | ---: |
| 2023 | 27,20 | 0,2720 |
| 2024 | 32,14 | 0,3214 |

Fontes primárias, consultadas em 30/10/2026:

- [ElCom, comunicado de 05/09/2023: Weiter steigende Strompreise 2024](https://www.elcom.admin.ch/de/nsb?id=97619).
  O comunicado informa 32,14 centavos para 2024 e aumento de 4,94 centavos.
- [Texto integral do mesmo comunicado no portal federal ComCom](https://www.comcom.admin.ch/de/nsb?id=97619).
  Explicita os 27,20 centavos de 2023 usados na comparação. Essa referência
  atualizada substitui, neste cenário, os 26,95 centavos anunciados originalmente
  em setembro de 2022 para 2023.
- [ElCom, relatório de atividades 2023](https://www.elcom.admin.ch/dam/en/sd-web/oUBbU5dWqAZR/taetigkeitsbericht2023.pdf),
  páginas impressas 50–51, seção 5.5 e figura 10: componentes tarifários e
  apresentação dos preços sem IVA.

Trata-se de um benchmark residencial nacional aplicado às cargas agregadas do
feeder, sem afirmar que essa rede tem um contrato H4 ou uma distribuidora
específica. A tarifa fica constante em cada ano; não representa preços spot,
nem contém tarifas artificiais de ponta e fora de ponta. Não foram acrescentadas
parcelas fixas de uma fatura ou IVA.

## Perfil fotovoltaico

Como a localização exata das cargas não foi fornecida, adotou-se **Zurique**
como referência de cenário: latitude 47,3769°, longitude 8,5417°, altitude
assumida de 408 m, fuso `Europe/Zurich`. Esses parâmetros são hipóteses de
modelagem, não metadados inferidos de `demand.csv`.

Fonte meteorológica: [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api),
com seleção explícita de **ECMWF ERA5** para ambos os anos. O resultado bruto
está em `sources/open_meteo_era5.json`; a URL completa, os parâmetros e a data
da consulta estão em `sources/weather_request.json`. A célula retornada fica
em 47,5° N, 8,5° E. Foram obtidas temperatura a 2 m, vento a 10 m em m/s e
radiação global, direta horizontal e difusa em W/m². Atribuição: Open-Meteo e
ECMWF/Copernicus ERA5; dados disponibilizados pela API sob
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), conforme a
[especificação oficial da API](https://github.com/open-meteo/open-meteo/blob/main/openapi/historical-weather.yml).

A radiação horária refere-se à **hora anterior** ao timestamp meteorológico.
O gerador respeita esse deslocamento e distribui as componentes direta e difusa
usando pesos de céu claro do pvlib. Calcula potência em pontos médios de cinco
minutos e tira a média de três amostras por intervalo de demanda. As médias
horárias de irradiância são preservadas. A resolução de 15 minutos é modelada
a partir da reanálise horária; não representa medições meteorológicas a cada
15 minutos.

Foi usado **pvlib 0.16.1**, com parâmetros explícitos no gerador:

- Arranjo de 260 kWp, igual ao `p_max_kw` do dispositivo existente; limite AC
  de 260 kW, respeitando o inversor de 300 kVA cadastrado em `devices.json`.
- Orientação sul, azimute 180° na convenção pvlib, inclinação de 35° e albedo 0,2.
- Posição solar pvlib e céu claro Ineichen para a distribuição temporal;
  transposição [Hay-Davies](https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.irradiance.get_total_irradiance.html).
- Temperatura [Faiman](https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.temperature.faiman.html),
  com `u0=25` e `u1=6.84`; o vento a 10 m é usado como aproximação do vento no módulo.
- Potência DC [PVWatts](https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.pvsystem.pvwatts_dc.html),
  coeficiente térmico de -0,004/°C e perda global DC assumida de 14%.
- Conversão AC pelo [inversor PVWatts](https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.inverter.pvwatts.html),
  eficiência nominal de 96%; potência nula durante a noite.

O perfil é uma simulação com meteorologia histórica, não uma série medida de
uma usina suíça. Não há modelagem separada de neve, obstáculos, sujeira variável,
perdas espectrais ou perdas por ângulo de incidência; a perda global é uma
aproximação agregada. As perdas associadas ao controle de reativos continuam
sob responsabilidade dos modelos dos dispositivos.

## Reprodução e uso no treinamento

Na raiz do repositório, com as dependências de `requirements.txt` instaladas:

```powershell
.\.venv\Scripts\python.exe -m tools.scripts.generate_profiles
```

O gerador reutiliza a resposta meteorológica salva e dispensa rede nas próximas
execuções. `profile_metadata.json` registra hipóteses, versões, parâmetros,
estatísticas anuais e hashes SHA-256 dos dados e do gerador.

`config.json` seleciona `single_phase_socp`. Para resolver o OPF, use horizontes
de treinamento limitados; os dois anos completos contêm 70.176 períodos.

**Sale-price scenario (2026-11-06):** `sell_price_per_kwh` in `price.csv`
is 20% of the unchanged purchase-price column `price_per_kwh`, in CHF/kWh.
This ratio is an experimental assumption, not a sourced Swiss feed-in tariff.
`grid.feed_in_tariff_ratio=0.2` applies the same relationship in the teacher
and environment; their loaders still calculate sale prices from this ratio.
The generator reproduces the new column using the configured ratio.
The former symmetric purchase/sale-price scenario used ratio 1.0; previously
saved training and test results retain that original economic assumption.

## Validação realizada em 30/10/2026

- Mesmos 70.176 timestamps e mesma ordem nos três CSVs; nenhum valor ausente
  ou não finito; PV entre zero e 260 kW. Irradiação horária conservada na
  desagregação, sem radiação descartada em horas inteiramente noturnas.
- `demand.csv` preservado, SHA-256
  `c6e6a9285bbeb411a9739980ebee821c43846eed578f4bb6d47a7905591f5a84`.
- Reprodução offline com hashes idênticos para demanda, preços, PV e metadados.
- Cenário carregado no teacher e no ambiente com `dt_h=0.25`; observação do
  ambiente aceita por `Teacher.observe_dict`.
- SOCP/Gurobi resolvido em um horizonte inicial de 96 períodos; primeira ação
  executada no OpenDSS. Outro horizonte de oito períodos, iniciado em
  04/04/2023 11:00 UTC e incluindo o pico de PV, também resolveu como `optimal`.
  A primeira geração solicitada foi aproximadamente 226,921 kW e a executada
  226,920 kW.

No segundo teste, o diagnóstico foi `socp_tightness=not_tight`, com gap
normalizado máximo de aproximadamente `2.38415e-5`, acima da tolerância padrão
`1e-5`. O êxito do solver e da execução não assegura que todo rótulo tenha uma
relaxação suficientemente justa. A coleta para imitação deve verificar esse
diagnóstico antes de aceitar uma amostra. O modelo do teacher não foi alterado.
