# ITU P528.5 Implementation

This is an ITU P528.5 implementation based on the following resources:
- [NTIA p528](https://github.com/NTIA/p528)
- [eeveetza p528](https://github.com/eeveetza/p528)
- [ITU-R Study Groups](https://www.itu.int/en/ITU-R/study-groups/rsg3/Pages/iono-tropo-spheric.aspx)
- [ITU-R P528 C++](https://www.itu.int/en/ITU-R/study-groups/rsg3/Pages/iono-tropo-spheric.aspx)

- [José P528 C++](https://github.com/josefelipe0036/p528_article/tree/main)

```
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

| Variable          | Type   | Units | Limits       | Description  |
|-------------------|--------|-------|--------------|--------------|
| `d__km`               | scalar double | km   | 0 < `d`   | Great circle path distance between terminals  |
| `h_1__meter`      | scalar double | m    | 1.5 ≤ `h_1__meter` ≤ 20000 | Height of the low terminal |
| `h_2__meter`      | scalar double | m    | 1.5 ≤ `h_2__meter` ≤ 20000 | Height of the high terminal |
| `f__mhz`          | scalar double | MHz    | 100 ≤ `f__mhz` ≤ 30000   | Frequency|
| `T_pol`           | scalar int    |       |             |  Polarization <br> 0 = horizontal <br> 1 = vertical |
| `p`          | scalar double | %    | 1 ≤ `p` ≤ 99   | Time percentage|


| Variable   | Type   | Units | Description |
|------------|--------|-------|-------------|
| `A__db`    | double | dB    | Basic transmission loss |
| `d__km`	| double  |	km	|Great circle path distance. Could be slightly different than specified in input variable if within LOS region |
| `A_fs__db`    | double | dB    | Free-space basic transmission loss |
| `A_a__db`    | double | dB    | Median atmospheric absorption loss |
| `theta_h1__rad`    | double | rad    | Elevation angle of the ray at the low terminal|
| `propagation_mode`    | int |    | Mode of propagation <br>1 = Line of Sight<br> 2 = Diffraction<br> 3 = Troposcatter|
| `rtn`    | int |    | Return flags / Error codes|
