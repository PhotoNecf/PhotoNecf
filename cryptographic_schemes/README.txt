Please note that this project is only for research purpose.

Our project "imageFPZKP" is developed based on jsnark(https://github.com/akosba/jsnark), a java library for building circuits for preprocessing zk-SNARKs, in order to run our code, please:

1. Run "sh build.sh". It first installs jsnark, zksnark and prerequisites and then move our project codes to correct directory in jsnark.
2. If error occurs, please go to  https://github.com/akosba/jsnark for an explicit instruction.
3. In directory "JsnarkCircuitBuilder" run
"java -cp bin examples.generators.IOT.Corr1CircuitGenerator" or "java -cp bin examples.generators.IOT.Corr2CircuitGenerator" for proof and verification of the correctness of image correlation algorithm
"java -cp bin examples.generators.MyPoolingCircuitGenerator" for proof and verification of the correctness of pooling algorithm
"java -cp bin examples.generators.NCCCircuitGenerator" for proof and verification of the correctness of NCC algorithm
"java -cp bin examples.generators.SOBELCircuitGenerator" for proof and verification of the correctness of SOBEL algorithm

Please note that you will need to change the file path in NCCCircuitGenerator.java in line 46, 60, 73, 85, 98 and 111.
You may also need to change the property path in jsnark/JsnarkCircuitBuilder/config.properties in line 22.
 