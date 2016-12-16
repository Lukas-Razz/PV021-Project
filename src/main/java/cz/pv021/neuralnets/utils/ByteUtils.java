package cz.pv021.neuralnets.utils;

import java.util.Arrays;

/**
 * @author  Josef Plch
 * @since   2016-12-13
 * @version 2016-12-13
 */
public abstract class ByteUtils {
    public static double[] byteToOneHotVector (byte byte8) {
        double[] array = new double[256];
        Arrays.fill (array, -1.0);
        int oneIndex = byteToInt (byte8);
        array[oneIndex] = 1;
        return array;
    }
    
    public static int byteToInt (byte byte8) {
        return (byte8 & 0xFF);
    }
    
    public static boolean getBit (byte byte8, int bitIndex) {
        return (byte8 >> bitIndex & 1) == 1;
    }
}
