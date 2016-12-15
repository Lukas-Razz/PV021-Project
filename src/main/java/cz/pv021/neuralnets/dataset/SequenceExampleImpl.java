package cz.pv021.neuralnets.dataset;

import cz.pv021.neuralnets.utils.Pair;
import java.util.List;

/**
 * @param <C> Type of data class.
 * 
 * @author  Josef Plch
 * @since   2016-12-15
 * @version 2016-12-15
 */
public class SequenceExampleImpl <C extends DataClass> extends Pair <List <double[]>, C> implements SequenceExample <C> {
    public SequenceExampleImpl (List <double[]> attributes, C exampleClass) {
        super (attributes, exampleClass);
    }
    
    /**
     * Use getSequence() instead.
     * 
     * @return Example attributes.
     */
    @Deprecated
    @Override
    public List <double[]> getA () {
        return super.getA ();
    }
    
    @Override
    public List <double[]> getAttributes () {
        return super.getA ();
    }
    
    /**
     * Use getDataClass() instead.
     * 
     * @return Example class.
     */
    @Deprecated
    @Override
    public C getB () {
        return super.getB ();
    }
    
    @Override
    public C getExampleClass () {
        return super.getB ();
    }
}
