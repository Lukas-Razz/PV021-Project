package cz.pv021.neuralnets.utils;

/**
 * Representation of an network output and expected output
 * 
 * @author Lukas Daubner
 * @since   2016-11-17
 * @version 2016-11-17
 */
public class OutputExample {
    
    private double[] actualOutput;
    private double expectedOutput;

    public OutputExample(double[] actualOutput, double expectedOutput) {
        this.actualOutput = actualOutput;
        this.expectedOutput = expectedOutput;
    }

    public double[] getActualOutput() {
        return actualOutput;
    }

    public double getExpectedOutput() {
        return expectedOutput;
    }
}
