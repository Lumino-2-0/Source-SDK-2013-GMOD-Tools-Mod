# Source SDK 2013 — GMOD Tools X64 Mod
## Ce projet est en cours de développement, mais l'optimisation GPU/OpenCL est fonctionnelle !

Ça y est ! J'ai créé une version GPU de l'outil VVIS spécialement pour GMOD. Elle utilise OpenCL pour la compatibilité GPU et, pour une grande carte ouverte comme ma Kindercity (une grande ville avec de nombreux éléments à l'intérieur), la compilation avec ma carte graphique (RTX 4090) n'a pris que 2 secondes. Alors, essayez-la et profitez de cette fonctionnalité ! Je comparerai de nombreuses cartes avec la version originale de VVIS et ma version GPU afin d'identifier les différences et les points à optimiser. Mais pour l'instant, ça fonctionne, et le plus dur est fait.

### Voici toutes les informations :

Code source du **Source SDK 2013 GMOD (64 bits)** — outils de compilation optimisés pour **VBSP**, **VVIS** et **VRAD**, visant principalement l'amélioration des performances, l'accélération GPU et la compatibilité avec les compilateurs modernes.

Ce dépôt est une version expérimentale dérivée de la [version 2013 du SDK Source de Ficool2](https://github.com/ficool2/source-sdk-2013), adaptée spécifiquement pour **Garry’s Mod (64 bits)**.

Il sert d’**environnement de recherche et de test**, et non de version finale du SDK.

---

## Présentation du projet

Ce projet vise à explorer comment moderniser les outils de compilation classiques du moteur Source afin de tirer parti des compilateurs modernes, et même de l’accélération GPU via **OpenCL**.

Bien que ce dépôt contienne des outils entièrement compilables, il **n’est pas destiné aux utilisateurs finaux** qui souhaitent simplement des compilateurs améliorés.

Il a plutôt pour but de documenter les expérimentations, de tester les optimisations et de démontrer les évolutions potentielles du pipeline du SDK Source original.

---

## Information sur l’activité de développement

Ce dépôt (et d’autres sur mon compte) est actuellement maintenu à un rythme **très lent et irrégulier**. Le développement peut être **suspendu pendant de longues périodes** en raison d'un manque de priorité et de temps disponible.

Les mises à jour, les modifications ou les changements expérimentaux peuvent donc prendre **des mois, voire des années** avant d'être publiés.

L'objectif reste de préserver et de documenter mon travail au fil du temps, même si le développement actif est peu fréquent.

## État actuel

### VVIS_GPU (OpenCL)

La version accélérée par GPU de **VVIS** est **actuellement suspendue**.

Avec la sortie de **[VVIS++ et d'autres outils de Ficool2](https://ficool2.github.io/HammerPlusPlus-Website/tools.html)**, les calculs de visibilité basés sur le CPU sont déjà considérablement optimisés.

Par conséquent, la branche OpenCL reste un **concept expérimental**, qui sera réexaminé ultérieurement pour la recherche ou des tests hybrides CPU/GPU.

### VRAD
Des travaux sont en cours pour refactoriser et améliorer **VRAD** :

- Un nouveau **système de sortie détaillé** enregistrera les informations dans un fichier séparé. - Des options de ligne de commande supplémentaires sont prévues pour l'analyse et le débogage de l'éclairage.

- Exploration potentielle de l'éclairage indirect basé sur le GPU.

### VBSP
Des tests de profilage et de structure mineurs sont en cours pour analyser le comportement à la compilation et les éventuels goulots d'étranglement d'E/S.

---

## Objectifs

Ce dépôt n'est pas un kit de développement logiciel (SDK) téléchargeable ni un outil prêt à l'emploi pour remplacer les outils officiels.

Il est destiné aux développeurs et aux passionnés qui souhaitent :

- Comprendre comment moderniser et optimiser les outils Source.

- Expérimenter l'intégration d'OpenCL dans des bases de code C++ existantes.

- Étudier les performances à la compilation et en matière d'éclairage.

- Suivre l'évolution des tests, des expérimentations et des refactorisations au fil du temps.

Les prochaines mises à jour viseront également à **regrouper toutes les dépendances** dans un **unique exécutable** (par exemple, `vvis.exe`, `vrad.exe` sans DLL externes), garantissant ainsi une compatibilité simplifiée avec **Garry’s Mod**.

⚠️ **Remarque :**
Ce projet est pour l’instant **exclusivement compatible avec Windows**. La prise en charge de Linux n’est pas native et sera envisagée ultérieurement sur demande, avec des explications ou adaptations de compatibilité dédiées.

---

## Instructions de compilation (Windows)

### Configuration requise
- **Visual Studio 2022** avec :

- Développement d'applications de bureau en C++

- Outils de compilation MSVC v143

- Kit de développement logiciel (SDK) Windows 10 ou 11

- **CMake** (recommandé)

- **Kit de développement logiciel (SDK) OpenCL** (compatible Intel, AMD ou NVIDIA)

### Compilation
```bash
git clone https://github.com/<votre-nom-d'utilisateur>/SourceSDK2013-GMOD-64Bit.git
cd SourceSDK2013-GMOD-64Bit
cmake -B build -S . -A x64
cmake --build build --config Release
```
Les binaires compilés se trouveront dans :

```
/bin/vvis_GPU.exe (Exemple)
```
### ! Attention !

Pour que l'outil fonctionne, vous aurez besoin du sous-dossier et de la DLL qu'il contient : `/bin/bin/x64/filesystem_stdio.dll`.

Concernant `filesystem_stdio.dll`, vous devrez conserver `tier0.dll` et `vstdlib.dll` dans le même répertoire que l'exécutable ou la version Release.

## Exemple d'utilisation :

## L'utilisation de cet outil est identique à celle des outils originaux.

### (VVIS GPU) Configuration globale :

```
CheminVersVotreClone\bin\vvis_GPU.exe [OptionsSpécifiques] -game "[CheminVersLeDossierDuJeu]" "[CheminVersLeFichierBsp]"
```
### (VVIS GPU) Exemple spécifique :

```
.\vvis_GPU.exe -threads 24 -game "C:\Program Files (x86)\Steam\steamapps\common\GarrysMod\garrysmod" "C:\Program Files (x86)\Steam\steamapps\common\GarrysMod\garrysmod\map\gm_kindercity"
```

## Références et ressources

- [Valve Developer Wiki — Source SDK 2013](https://developer.valvesoftware.com/wiki/Source_SDK_2013)

- [Source SDK 2013 de Ficool2 (Projet de base)](https://github.com/ficool2/source-sdk-2013)

- [Outils de Ficool2 (VVIS++, VRAD++, etc.)](https://ficool2.github.io/HammerPlusPlus-Website/tools.html)

- [Spécifications OpenCL](https://www.khronos.org/opencl/)

- [Garry’s Mod Developer Wiki](https://wiki.facepunch.com/gmod/)

---

## Licence

Le SDK et ses dérivés sont distribués sous la **LICENCE SOURCE 1 SDK**, incluse dans le fichier [LICENSE](LICENSE). Toutes les modifications, expérimentations et codes dérivés pour GPU ou optimisation sont soumis aux mêmes conditions de licence non commerciale.

---

## Notes de l'auteur

Ce dépôt marque le **début d'un projet expérimental à long terme** visant à moderniser, analyser et optimiser les outils de compilation Source pour Garry's Mod et d'autres jeux.

L'accent est mis sur la **recherche**, l'**apprentissage** et la **curiosité technique** plutôt que sur la fourniture de binaires prêts à l'emploi.

> « Ce n'est que le début d'un long voyage d'expérimentations et d'idées. »
