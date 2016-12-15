package cz.pv021.neuralnets.dataset.udcorpora;

import cz.pv021.neuralnets.dataset.SequenceExampleImpl;
import java.util.List;

/**
 * @author  Josef Plch
 * @since   2016-12-15
 * @version 2016-12-15
 */
public class UdExample extends SequenceExampleImpl <UdLanguage> {
    public UdExample (List <double[]> attributes, UdLanguage exampleClass) {
        super (attributes, exampleClass);
    }
}
