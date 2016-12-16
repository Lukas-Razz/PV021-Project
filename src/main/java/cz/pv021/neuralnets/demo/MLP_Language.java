package cz.pv021.neuralnets.demo;

import cz.pv021.neuralnets.dataset.DataSet;
import cz.pv021.neuralnets.dataset.iris.IrisClass;
import cz.pv021.neuralnets.dataset.udcorpora.UdLanguage;
import cz.pv021.neuralnets.dataset.udcorpora.UdSimpleExample;
import cz.pv021.neuralnets.optimizers.*;
import cz.pv021.neuralnets.error.*;
import cz.pv021.neuralnets.initialization.*;
import cz.pv021.neuralnets.layers.*;
import cz.pv021.neuralnets.functions.*;
import cz.pv021.neuralnets.network.MultilayerPerceptron;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileSystems;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import cz.pv021.neuralnets.utils.ModelStatistics;
import cz.pv021.neuralnets.utils.OutputExample;
import java.io.IOException;
import static java.lang.Math.min;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedList;

/**
 * @author  Lukáš Daubner
 * @since   2016-10-30
 * @version 2016-12-14
 */
public class MLP_Language {
    private static final Logger LOGGER = LoggerFactory.getLogger (MLP_Language.class);
        
    public static void main (String[] args) {
        try {
            run();
        }
        catch (IOException exception) {
            LOGGER.error ("IO failed.", exception);
        }
    }
    
    private static void run () throws IOException {
        double learningRate = 0.001;
        double momentum = 0.01;
        double l1 = 0.00;
        double l2 = 0.000;
        
        int bachSize = 1;
        int epochs = 10;
        
        Cost cost = new Cost (new RootMeanSquaredError(), l1, l2);
        Optimizer optimizer = new Optimizer(learningRate, new SGD(), momentum, l1, l2);
        
        InputLayer  layer0 = new InputLayerImpl (0, 100);
        HiddenLayer layer1a = new FullyConnectedLayer (1, 150, new HyperbolicTangent());
        //HiddenLayer layer1b = new FullyConnectedLayer (2, 10, new HyperbolicTangent());
        OutputLayer layer2  = new OutputLayerImpl (2, UdLanguage.size (), new Softmax ());
        
        Initializer initializer = new Initializer (new NormalInitialization (123456));
        
        MultilayerPerceptron <InputLayer, OutputLayer> mlp = new MultilayerPerceptron <> (
            Arrays.asList (layer0),
            Arrays.asList (layer1a),
            layer2,
            cost,
            optimizer
        );
        mlp.initializeWeights (initializer);
        
        int classes = UdLanguage.size ();
        int[][] confusionMatrix = new int[classes][classes];
        for (int row = 0; row < classes; row++) {
            for (int col = 0; col < classes; col++) {
                confusionMatrix[row][col] = 0;
            }
        }
        
        int minimalLength = 75; //Remove outliers
        int maximumLength = 100;
        List <UdSimpleExample> czExamples = parseLanguageFile("cs_sentences", UdLanguage.CZECH, minimalLength, maximumLength);
        List <UdSimpleExample> deExamples = parseLanguageFile("de_sentences", UdLanguage.GERMAN, minimalLength, maximumLength);
        List <UdSimpleExample> enExamples = parseLanguageFile("en_sentences", UdLanguage.ENGLISH, minimalLength, maximumLength);
        List <UdSimpleExample> esExamples = parseLanguageFile("es_sentences", UdLanguage.SPANISH, minimalLength, maximumLength);
        List <UdSimpleExample> frExamples = parseLanguageFile("fr_sentences", UdLanguage.FRENCH, minimalLength, maximumLength);
        List <UdSimpleExample> huExamples = parseLanguageFile("hu_sentences", UdLanguage.HUNGARIAN, minimalLength, maximumLength);
        List <UdSimpleExample> itExamples = parseLanguageFile("it_sentences", UdLanguage.ITALIAN, minimalLength, maximumLength);
        List <UdSimpleExample> laExamples = parseLanguageFile("la_sentences", UdLanguage.LATIN, minimalLength, maximumLength);
        List <UdSimpleExample> plExamples = parseLanguageFile("pl_sentences", UdLanguage.POLISH, minimalLength, maximumLength);
        
        Integer minimumOfExamples = 
                min(czExamples.size(),
                min(deExamples.size(), 
                min(enExamples.size(),
                min(esExamples.size(),
                min(frExamples.size(),
                min(huExamples.size(),
                min(itExamples.size(),
                min(laExamples.size(), plExamples.size()))))))));
        
        List<UdSimpleExample> allExamples = new LinkedList<>();
        for(int i=0; i<minimumOfExamples; i++) {
            allExamples.add(czExamples.get(i));
            allExamples.add(deExamples.get(i));
            allExamples.add(enExamples.get(i));
            allExamples.add(esExamples.get(i));
            allExamples.add(frExamples.get(i));
            allExamples.add(huExamples.get(i));
            allExamples.add(itExamples.get(i));
            allExamples.add(laExamples.get(i));
            allExamples.add(plExamples.get(i));
        }
        
        DataSet<UdLanguage, UdSimpleExample> dataSet = new DataSet<>(allExamples, 0.2);
        dataSet.normalizeToMinusOnePlusOne();
        /*
        for(UdSimpleExample e : dataSet.getTrainSet()) {
            LOGGER.info(e.toString());
        }
        */
        LOGGER.info(String.valueOf(dataSet.getTrainSet().size()));
        LOGGER.info(String.valueOf(dataSet.getTestSet().size()));
            
        
        List<List<UdSimpleExample>> batches = dataSet.splitToBatch(bachSize);
        for (int epoch = 0; epoch < epochs; epoch++) {
            runEpoch (mlp, batches, epoch);
        }

        test(mlp, dataSet.getTestSet());
    }
    
    private static void runEpoch (MultilayerPerceptron <?, ?> mlp, List<List<UdSimpleExample>> batches, int epoch) {
        // Set up the confusion matrix.
        int classes = UdLanguage.values().length;
        int[][] confusionMatrix = new int[classes][classes];
        for (int row = 0; row < classes; row++) {
            for (int col = 0; col < classes; col++) {
                confusionMatrix[row][col] = 0;
            }
        }

        double error = 0;
        for(List<UdSimpleExample> batch : batches) {
            List<OutputExample> batchOutput = new ArrayList<>();
            for (UdSimpleExample example : batch) {
                double[] attributes = example.getAttributes ();
                int correctClassNumber = example.getExampleClass ().getIndex ();
                
                mlp.setInputs (Collections.singletonList (attributes));
                mlp.forwardPass ();
                mlp.setExpectedOutput (correctClassNumber);
                mlp.backwardPass ();

                batchOutput.add(new OutputExample(mlp.getOutput(), correctClassNumber));
                //LOGGER.info(Arrays.toString(mlp.getOutput()));
                int predictedClassNumber = mlp.getOutputClassIndex ();
                confusionMatrix[correctClassNumber][predictedClassNumber]++;
            }
            error += mlp.getBatchError(batchOutput);
            //LOGGER.info(String.valueOf(mlp.getBatchError(batchOutput)));
            mlp.adaptWeights ();
        }
        error = error / batches.size();
        
        System.out.println ("Epoch #" + epoch + ": average error = " + error);
    }
    
    private static void test (MultilayerPerceptron <?, ?> mlp, List<UdSimpleExample> testSet) {
        // Set up the confusion matrix.
        int classes = UdLanguage.values().length;
        int[][] confusionMatrix = new int[classes][classes];
        for (int row = 0; row < classes; row++) {
            for (int col = 0; col < classes; col++) {
                confusionMatrix[row][col] = 0;
            }
        }
         
        for (UdSimpleExample example : testSet) {
            double[] attributes = example.getAttributes ();
            int correctClassNumber = example.getExampleClass ().getIndex ();

            mlp.setInputs (Collections.singletonList (attributes));
            mlp.forwardPass ();

            int predictedClassNumber = mlp.getOutputClassIndex ();
            confusionMatrix[correctClassNumber][predictedClassNumber]++;
        }
        
        System.out.println ();
        System.out.println ("=== Model test results ===");
        String statistics = ModelStatistics.modelStatistics (confusionMatrix, UdLanguage.values ());
        System.out.println (statistics);
        System.out.println ();
     }
    
    private static List<UdSimpleExample> parseLanguageFile(String path, UdLanguage language , int minLength, int maxLength) throws IOException {
        List <UdSimpleExample> examples = new LinkedList <> ();
        String DataPath = "./data/language_identification/" + path + ".txt";
        Path DataFilePath = FileSystems.getDefault().getPath (DataPath);
        List <String> sentences = Files.readAllLines (DataFilePath, StandardCharsets.UTF_8);
        for (String sentence : sentences) {
            byte[] bytes = sentence.getBytes (StandardCharsets.UTF_8);
            double[] dBytes = new double[maxLength];
            
            if(bytes.length >= minLength) {
                for(int b=0; b<bytes.length; b++) {
                    if(b == maxLength-1){
                        break;
                    }
                    dBytes[b] = (double)bytes[b];
                }
                examples.add(new UdSimpleExample(dBytes, language));
            }
        }
        return examples;
    }
}
