# Code licenses and corresponding source

The original project components remain available under the [MIT license](LICENSE).
The combined two-engine Streamlit application, including `streamlit_app.py`,
`keyclf/cloud_audio.py`, and `keyclf/cloud_engines.py`, is offered under the
[GNU Affero General Public License, version 3](LICENSE-AGPL). When these
components run together, the corresponding build and application source is
available in the [project repository](https://github.com/AreteKaiAndreia/Key_Signature).
The app links to the build instructions; use the deployed release's commit
when publishing a public build.

The Essentia engine uses the
[Essentia 2.1b6.dev1389 wheel](https://pypi.org/project/essentia/2.1b6.dev1389/),
licensed by the Music Technology Group, Universitat Pompeu Fabra. The
[upstream source at its recorded revision](https://github.com/MTG/essentia/tree/36ec3d92),
[licensing information](https://essentia.upf.edu/licensing_information.html),
and [authors notice](ESSENTIA-AUTHORS) are provided here. Build instructions
are in [DEPLOYMENT.md](DEPLOYMENT.md).

Neither code license grants rights to upstream audio, labels, feature caches, or
trained artifacts. See [DATA_PROVENANCE.md](DATA_PROVENANCE.md).
