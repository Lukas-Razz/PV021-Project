package cz.pv021.neuralnets.dataset.iris;

import cz.pv021.neuralnets.dataset.DataClass;
import java.util.Objects;

/**
 * Classes of Iris data set.
 * 
 * @author  Josef Plch
 * @since   2016-12-03
 * @version 2016-12-10
 */
public enum IrisClass implements DataClass {
    SETOSA ("Iris-setosa"),
    VERSICOLOR ("Iris-versicolor"),
    VIRGINICA ("Iris-virginica");
    private final String code;
    
    IrisClass (String code) {
        this.code = code;
    }
    
    @Override
    public int getIndex () {
        return this.ordinal ();
    }
    
    @Override
    public String getName () {
        return this.name ();
    }
    
    public static IrisClass read (String string) {
        for (IrisClass irisClass : IrisClass.values()) {
            if (Objects.equals (string, irisClass.code)) {
                return irisClass;
            }
        }
        throw new IllegalArgumentException ("Invalid class code.");
    }
}
