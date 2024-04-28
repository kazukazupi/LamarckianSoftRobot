# Lamarckian Soft Robot

## 概要
[Genetic and Evolutionary Computation Conference (GECCO-2024)](http://gecco-2024.sigevo.org/HomePage)に採択された論文「Lamarckian Co-design of Soft Robots via Transfer Learning」の実装のgithubレポジトリです。
深層強化学習による学習結果を個体間で共有することで、より効率の良いソフトロボットの設計を可能にしました。ロボットのシミュレーションには[evogym](https://evolutiongym.github.io/)を用いています。

## 環境構築

### evogymの環境構築
[evogymのgithubレポジトリ](https://github.com/EvolutionGym/evogym)のInstallationに従い、evogymを動かすための環境構築を行なってください。condaを用いた場合でしか検証を行っていないので、condaを用いる子をと推奨します。

### 本レポジトリのための環境構築
このレポジトリをローカルにcloneしてください。

その上でevogym環境に入ります。
```bash
conda activate evogym
```
本レポジトリに特有のモジュールを以下のようにインストールします。

```bash
conda install pydantic
```

## Pull Requestを出す場合
Pull Requestを出す場合は、フォーマッタ・リンターもインストールしてください:

```bash
conda install isort
conda install black
conda install mypy
```

フォーマッタ・リンターをかけてからPull Requestを出してください:
```bash
isort ./
balck ./
mypy ./
```