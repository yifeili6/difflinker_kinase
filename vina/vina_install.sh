# Install vina
RUN wget https://vina.scripps.edu/wp-content/uploads/sites/55/2020/12/autodock_vina_1_1_2_linux_x86.tgz
RUN tar -xf autodock_vina_1_1_2_linux_x86.tgz
RUN mv /autodock_vina_1_1_2_linux_x86/bin/vina /usr/local/bin/vina

# Install MGTools (necessary for running vina and autodock)
RUN wget https://ccsb.scripps.edu/mgltools/download/495/MGLTools-1.5.6-Linux-x86_64.tar.gz
RUN tar -xf MGLTools-1.5.6-Linux-x86_64.tar.gz
RUN cd mgltools_x86_64Linux2_1.5.6 && ./install.sh
RUN echo "export PATH=\$PATH:/mgltools_x86_64Linux2_1.5.6/bin" >> $HOME/.bashrc
RUN echo "export PATH=\$PATH:/mgltools_x86_64Linux2_1.5.6/MGLToolsPckgs/AutoDockTools/Utilities24" >> $HOME/.bashrc
