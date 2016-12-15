package cz.pv021.neuralnets.layers;

import cz.pv021.neuralnets.utils.Pair;
import java.util.Collections;
import java.util.List;

/**
 * @author  Josef Plch
 * @since   2016-12-13
 * @version 2016-12-15
 */
public class InputMerger {
    private final List <LayerWithOutput> layers;
    private final int numberOfUnits;

    public InputMerger (List <LayerWithOutput> layers) {
        this.layers = layers;
        int sum = 0;
        for (LayerWithOutput layer : layers) {
            sum += layer.getNumberOfUnits ();
        }
        this.numberOfUnits = sum;
    }

    private Pair <Integer, Integer> convertIndex (int index) {
        Integer layerIndex = 0;
        Integer neuronIndex = null;

        for (LayerWithOutput layer : layers) {
            int layerSize = layer.getNumberOfUnits ();
            if (index <= layerSize) {
                neuronIndex = index;
                break;
            }
            else {
                index -= layerSize;
                layerIndex++;
            }
        }

        if (neuronIndex == null) {
            throw new IllegalArgumentException ("Given index is out of bounds.");
        }
        else {
            return new Pair <> (layerIndex, neuronIndex);
        }
    }
    
    /**
     * @return Read-only list of layers.
     */
    public List <LayerWithOutput> getLayers () {
        return Collections.unmodifiableList (layers);
    }

    public int getNumberOfUnits () {
        return numberOfUnits;
    }

    public double[] getOutput () {
        double[] output;
        if (layers.size () == 1) {
            output = layers.get(0).getOutput();
        }
        else {
            output = new double[numberOfUnits];
            int index = 0;
            for (LayerWithOutput layer : layers) {
                double[] layerOutput = layer.getOutput ();
                int layerSize = layer.getNumberOfUnits ();
                System.arraycopy (layerOutput, 0, output, index, layerSize);
                index += layerSize;
            }
        }
        return output;
    }

    public double getOutput (int index) {
        Pair <Integer, Integer> convertedIndex = convertIndex (index);
        int i1 = convertedIndex.getA ();
        int i2 = convertedIndex.getB ();
        return layers.get(i1).getOutput()[i2];
    }

    public void setOutput (int index, double value) {
        Pair <Integer, Integer> convertedIndex = convertIndex (index);
        int i1 = convertedIndex.getA ();
        int i2 = convertedIndex.getB ();
        layers.get(i1).getOutput()[i2] = value;
    }
}
