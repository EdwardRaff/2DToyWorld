/*
 * Copyright (C) 2014 Edward Raff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package com.edwardraff.toyworld;

import com.edwardraff.jsatfx.ClassificationPlot;
import com.edwardraff.jsatfx.Plot;
import com.edwardraff.jsatfx.swing.ParameterPanel;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.io.*;
import static java.lang.Double.parseDouble;
import java.util.*;
import java.util.List;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.logging.Level;
import java.util.logging.Logger;
import javafx.application.Platform;
import javafx.embed.swing.JFXPanel;
import javafx.scene.Scene;
import javafx.scene.layout.BorderPane;
import javax.swing.*;
import jsat.*;
import jsat.classifiers.*;
import jsat.classifiers.bayesian.*;
import jsat.classifiers.boosting.*;
import jsat.classifiers.knn.NearestNeighbour;
import jsat.classifiers.linear.LogisticRegressionDCD;
import jsat.classifiers.linear.PassiveAggressive;
import jsat.classifiers.linear.SPA;
import jsat.classifiers.neuralnetwork.BackPropagationNet;
import jsat.classifiers.neuralnetwork.Perceptron;
import jsat.classifiers.svm.DCDs;
import jsat.classifiers.svm.PlatSMO;
import jsat.classifiers.trees.*;
import jsat.datatransform.*;
import jsat.distributions.kernels.RBFKernel;
import jsat.distributions.multivariate.MetricKDE;
import jsat.linear.*;
import static jsat.linear.DenseVector.toDenseVec;
import jsat.parameters.*;
import jsat.utils.SystemInfo;

/**
 * GUI for visualizing classification 2D problems
 * 
 * @author Edward Raff
 */
@SuppressWarnings("serial")
public class ClassifierToyWorld extends javax.swing.JFrame
{
    /**
     * This will be the currently loaded dataset
     */
    private static ClassificationDataSet dataSet;
    final JFileChooser fileChooser = new JFileChooser();
    /**
     * The main holder of all the visualizations. Empty until a dataset is 
     * loaded. The first index will be a visualization of the dataset, all 
     * subsequent indices are classifiers we have trained 
     */
    private static JTabbedPane centerTabbed;
    /**
     * Atomic integer keeps track of the number of classifiers we are currently 
     * waiting to finish training
     */
    private static final AtomicInteger waitingFor = new AtomicInteger(0);
    /**
     * Queue of classification jobs to run
     */
    private static BlockingQueue<Runnable> backgroundJobQueue;
    /**
     * The thread we use to eat classification jobs forever
     */
    private static Thread backgroundThread;
    /**
     * List of all plots so that when settings are changed we can apply them to 
     * all the current plots
     */
    private static final List<ClassificationPlot> plotList = new ArrayList<>();
    private static final ExecutorService execService = Executors.newFixedThreadPool(SystemInfo.LogicalCores);
    /**
     * Current resolutions for visualizing classification space. 
     */
    private static int plotResolution = 5;
    /**
     * Menu to hold all the transformation options
     */
    private static TransformsMenu transformsMenu;
    /**
     * Indicates whether we want to give classes equal weight when training the 
     * model or not
     */
    private static volatile boolean equalWeight = false;
    
    /**
     * The list of all classifiers we know of
     */
    private static final Map<String, Classifier> classifierss = new LinkedHashMap<String, Classifier>()
    {{
        put("k-NearestNeighbour", new NearestNeighbour(1));
        put("MetricKDE", new BestClassDistribution(new MetricKDE(), true));
        put("Decision Stump", new DecisionStump());
        put("Decision Tree", new DecisionTree());
        put("RandomForest", new RandomForest(100));
        put("PassiveAggressive", new PassiveAggressive());
        put("SPA", new SPA());
        put("Preceptron", new Perceptron());
        put("Naive Bayes", new NaiveBayes(NaiveBayes.NumericalHandeling.NORMAL));
        put("Multivariate Normals", new MultivariateNormals(true));
        put("Logistic Regression", new LogisticRegressionDCD());
        put("SVM-Linear", new DCDs());
        put("SVM-RBF", new PlatSMO(new RBFKernel(0.075)));
        put("Back Propagation Net", new BackPropagationNet(new int[]{10}));
        
    }};
    
    /**
     * Creates new form ClassifierToyWorld
     */
    public ClassifierToyWorld()
    {
        initComponents();
        jMenuBar1.add(transformsMenu = new TransformsMenu(this, "Transforms"));
        
        jLabelInfo.setText(" ");
        backgroundJobQueue = new LinkedBlockingQueue<>();
        //add a menu item fro every classifier we have
        for(Map.Entry<String, Classifier> entry : classifierss.entrySet())
        {
            final String name = entry.getKey();
            final Classifier classifier = entry.getValue();
            
            JMenuItem menuItem = new JMenuItem(name);
            menuItem.addActionListener((ActionEvent ae) ->
            {
                jLabelInfo.setText("Waiting on " + waitingFor.incrementAndGet() + " jobs...");
                try
                {
                    //First, configure out objects and ask the User to change settings. 
                    showParameterizedDialog(getOwner(), classifier);
                    Classifier workingClassifier = classifier.clone();
                    if (!jRadioBMetaNone.isSelected())
                    {
                        if (jRadioBMetaBagging.isSelected())
                            workingClassifier = new Bagging(workingClassifier);
                        else if (jRadioBMetaAdaBoosM1.isSelected())
                            workingClassifier = new AdaBoostM1(workingClassifier, 100);
                        else if (jRadioBMetaSAMME.isSelected())
                            workingClassifier = new SAMME(workingClassifier, 100);
                        else if (jRadioBMetaEmphasisBoost.isSelected())
                            workingClassifier = new EmphasisBoost(workingClassifier, 100, 0.5);
                        else if (jRadioBMetaModestBoost.isSelected())
                            workingClassifier = new ModestAdaBoost(workingClassifier, 100);

                        showParameterizedDialog(getOwner(), workingClassifier);
                    }
                    String prefix;
                    if (jRadioButtonMenuItemMultiClassOneVsAll.isSelected())
                    {
                        prefix = "One-vs-All " ;
                        workingClassifier = new OneVSAll(workingClassifier, false);
                    }
                    else if (jRadioButtonMenuItemMultiClassOneVsOne.isSelected())
                    {
                        prefix = "One-vs-One " ;
                        workingClassifier = new OneVSOne(workingClassifier, false);
                    }
                    else if (jRadioButtonMenuItemMultiClassDDAG.isSelected())
                    {
                        prefix = "DDAG " ;
                        workingClassifier = new DDAG(workingClassifier, false);
                    }
                    else
                        prefix = "";
                    workingClassifier = new DataModelPipeline(workingClassifier, transformsMenu.getDataTransformProcess().clone());
                    
                    //make the reference final so we can just call it below in the lambda
                    final Classifier finalClassifier = workingClassifier;
                        
                    //and now queue it to run in the background
                    backgroundJobQueue.put((Runnable) () ->
                    {
                        
                        if (equalWeight)
                        {
                            double[] priors = dataSet.getPriors();
                            for (int i1 = 0; i1 < dataSet.getSampleSize(); i1++)
                                dataSet.getDataPoint(i1).setWeight(1.0 - priors[dataSet.getDataPointCategory(i1)]);
                        }
                        else
                            for (int i2 = 0; i2 < dataSet.getSampleSize(); i2++)
                                dataSet.getDataPoint(i2).setWeight(1.0);
                        
                        try
                        {
                            if(jCheckBoxMenuItemParallel.isSelected())
                                finalClassifier.trainC(dataSet, execService);
                            else
                                finalClassifier.trainC(dataSet);
                        }
                        catch(final Exception ex)
                        {
                            SwingUtilities.invokeLater(() ->
                            {
                                JOptionPane.showMessageDialog(rootPane, "Error: " + ex.getMessage(), "Error ", JOptionPane.ERROR_MESSAGE);
                            });
                            
                            return;
                        }
                        
                        ClassificationPlot cp = Plot.classification(dataSet, finalClassifier);
                        cp.setResolution(plotResolution);
                        cp.setHardBoundaries(jCheckBoxMenuItemHardBoundaries.isSelected());
                        plotList.add(cp);
                        
                        final JFXPanel fxPanel = new JFXPanel();
                        Platform.runLater(() ->
                        {
                            fxPanel.setScene(new Scene(new BorderPane(cp)));
                        });
                        centerTabbed.add(prefix + name, fxPanel);
                    });
                }
                catch (InterruptedException ex)
                {
                    Logger.getLogger(ClusterToyWorld.class.getName()).log(Level.SEVERE, null, ex);
                }
            });
            
            jMenuClassifiers.add(menuItem);
        }
        
        
        //loop forever, eating jobs and updating the counter
        backgroundThread = new Thread(() -> 
        {
            while(true)
            {
                try
                {
                    Runnable toRun = backgroundJobQueue.take();
                    jLabelInfo.setVisible(true);
                    validate();
                    repaint();
                    toRun.run();
                    int now = waitingFor.decrementAndGet();
                    if(now > 0)
                        jLabelInfo.setText("Waiting on " + now + " jobs...");
                    else
                        jLabelInfo.setText(" ");
                }
                catch (InterruptedException ex)
                {
                    Logger.getLogger(ClusterToyWorld.class.getName()).log(Level.SEVERE, null, ex);
                }
            }
        });
        backgroundThread.setDaemon(true);
        backgroundThread.start();
        
        setSize(600, 400);
    }

    public static void showParameterizedDialog(Window window, Object classifier)
    {
        if (classifier instanceof Parameterized)
        {
            ParameterPanel pp = new ParameterPanel((Parameterized) classifier);
            final JDialog jd = new JDialog(window, "Set Parameters", Dialog.ModalityType.APPLICATION_MODAL);
            jd.setContentPane(pp);
            pp.getjButtonOk().addActionListener((ActionEvent e) ->
            {
                jd.setVisible(false);
            });
            jd.pack();
            jd.setVisible(true);
        }
    }

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents()
    {

        buttonGroupMultiClass = new javax.swing.ButtonGroup();
        buttonGroupMeta = new javax.swing.ButtonGroup();
        bottomAreaPanel = new javax.swing.JPanel();
        jLabelInfo = new javax.swing.JLabel();
        jMenuBar1 = new javax.swing.JMenuBar();
        jMenuFile = new javax.swing.JMenu();
        jMenuItemOpen = new javax.swing.JMenuItem();
        jMenuEdit = new javax.swing.JMenu();
        jMenuMultiClassExtension = new javax.swing.JMenu();
        jRadioButtonMenuItemMultiClassNative = new javax.swing.JRadioButtonMenuItem();
        jRadioButtonMenuItemMultiClassOneVsAll = new javax.swing.JRadioButtonMenuItem();
        jRadioButtonMenuItemMultiClassOneVsOne = new javax.swing.JRadioButtonMenuItem();
        jRadioButtonMenuItemMultiClassDDAG = new javax.swing.JRadioButtonMenuItem();
        jMenuCC = new javax.swing.JMenu();
        jRadioBMetaNone = new javax.swing.JRadioButtonMenuItem();
        jRadioBMetaBagging = new javax.swing.JRadioButtonMenuItem();
        jRadioBMetaAdaBoosM1 = new javax.swing.JRadioButtonMenuItem();
        jRadioBMetaSAMME = new javax.swing.JRadioButtonMenuItem();
        jRadioBMetaEmphasisBoost = new javax.swing.JRadioButtonMenuItem();
        jRadioBMetaModestBoost = new javax.swing.JRadioButtonMenuItem();
        jMenuItemPlotResolition = new javax.swing.JMenuItem();
        jCheckBoxMenuItemHardBoundaries = new javax.swing.JCheckBoxMenuItem();
        jCheckBoxMenuItemParallel = new javax.swing.JCheckBoxMenuItem();
        jCheckBoxMenuItemEqualWeight = new javax.swing.JCheckBoxMenuItem();
        jMenuClassifiers = new javax.swing.JMenu();

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);

        bottomAreaPanel.setBorder(javax.swing.BorderFactory.createEtchedBorder());
        bottomAreaPanel.setLayout(new java.awt.FlowLayout(java.awt.FlowLayout.RIGHT));

        jLabelInfo.setHorizontalAlignment(javax.swing.SwingConstants.CENTER);
        jLabelInfo.setText("This text will change");
        bottomAreaPanel.add(jLabelInfo);

        getContentPane().add(bottomAreaPanel, java.awt.BorderLayout.PAGE_END);

        jMenuFile.setText("File");

        jMenuItemOpen.setText("Open");
        jMenuItemOpen.addActionListener(new java.awt.event.ActionListener()
        {
            public void actionPerformed(java.awt.event.ActionEvent evt)
            {
                jMenuItemOpenActionPerformed(evt);
            }
        });
        jMenuFile.add(jMenuItemOpen);

        jMenuBar1.add(jMenuFile);

        jMenuEdit.setText("Edit");

        jMenuMultiClassExtension.setText("Multi-Class Handling");

        buttonGroupMultiClass.add(jRadioButtonMenuItemMultiClassNative);
        jRadioButtonMenuItemMultiClassNative.setSelected(true);
        jRadioButtonMenuItemMultiClassNative.setText("Native");
        jMenuMultiClassExtension.add(jRadioButtonMenuItemMultiClassNative);

        buttonGroupMultiClass.add(jRadioButtonMenuItemMultiClassOneVsAll);
        jRadioButtonMenuItemMultiClassOneVsAll.setText("One-vs-All");
        jMenuMultiClassExtension.add(jRadioButtonMenuItemMultiClassOneVsAll);

        buttonGroupMultiClass.add(jRadioButtonMenuItemMultiClassOneVsOne);
        jRadioButtonMenuItemMultiClassOneVsOne.setText("One-vs-One");
        jMenuMultiClassExtension.add(jRadioButtonMenuItemMultiClassOneVsOne);

        buttonGroupMultiClass.add(jRadioButtonMenuItemMultiClassDDAG);
        jRadioButtonMenuItemMultiClassDDAG.setText("DDAG");
        jMenuMultiClassExtension.add(jRadioButtonMenuItemMultiClassDDAG);

        jMenuEdit.add(jMenuMultiClassExtension);

        jMenuCC.setText("Meta");

        buttonGroupMeta.add(jRadioBMetaNone);
        jRadioBMetaNone.setSelected(true);
        jRadioBMetaNone.setText("None");
        jMenuCC.add(jRadioBMetaNone);

        buttonGroupMeta.add(jRadioBMetaBagging);
        jRadioBMetaBagging.setText("Bagging");
        jMenuCC.add(jRadioBMetaBagging);

        buttonGroupMeta.add(jRadioBMetaAdaBoosM1);
        jRadioBMetaAdaBoosM1.setText("AdaBoostM1");
        jMenuCC.add(jRadioBMetaAdaBoosM1);

        buttonGroupMeta.add(jRadioBMetaSAMME);
        jRadioBMetaSAMME.setText("SAMME");
        jMenuCC.add(jRadioBMetaSAMME);

        buttonGroupMeta.add(jRadioBMetaEmphasisBoost);
        jRadioBMetaEmphasisBoost.setText("EmphasisBoost");
        jMenuCC.add(jRadioBMetaEmphasisBoost);

        buttonGroupMeta.add(jRadioBMetaModestBoost);
        jRadioBMetaModestBoost.setText("ModestBoost");
        jMenuCC.add(jRadioBMetaModestBoost);

        jMenuEdit.add(jMenuCC);

        jMenuItemPlotResolition.setText("Plot Resolution");
        jMenuItemPlotResolition.addActionListener(new java.awt.event.ActionListener()
        {
            public void actionPerformed(java.awt.event.ActionEvent evt)
            {
                jMenuItemPlotResolitionActionPerformed(evt);
            }
        });
        jMenuEdit.add(jMenuItemPlotResolition);

        jCheckBoxMenuItemHardBoundaries.setSelected(true);
        jCheckBoxMenuItemHardBoundaries.setText("Hard Boundaries");
        jCheckBoxMenuItemHardBoundaries.addActionListener(new java.awt.event.ActionListener()
        {
            public void actionPerformed(java.awt.event.ActionEvent evt)
            {
                jCheckBoxMenuItemHardBoundariesActionPerformed(evt);
            }
        });
        jMenuEdit.add(jCheckBoxMenuItemHardBoundaries);

        jCheckBoxMenuItemParallel.setText("Parallel Computation");
        jMenuEdit.add(jCheckBoxMenuItemParallel);

        jCheckBoxMenuItemEqualWeight.setText("Give Classes Equal Weight");
        jCheckBoxMenuItemEqualWeight.addActionListener(new java.awt.event.ActionListener()
        {
            public void actionPerformed(java.awt.event.ActionEvent evt)
            {
                jCheckBoxMenuItemEqualWeightActionPerformed(evt);
            }
        });
        jMenuEdit.add(jCheckBoxMenuItemEqualWeight);

        jMenuBar1.add(jMenuEdit);

        jMenuClassifiers.setText("Classifier");
        jMenuBar1.add(jMenuClassifiers);

        setJMenuBar(jMenuBar1);

        pack();
    }// </editor-fold>//GEN-END:initComponents

    private void jMenuItemOpenActionPerformed(java.awt.event.ActionEvent evt)//GEN-FIRST:event_jMenuItemOpenActionPerformed
    {//GEN-HEADEREND:event_jMenuItemOpenActionPerformed
        int returnVal = fileChooser.showOpenDialog(this);
        if (returnVal == JFileChooser.APPROVE_OPTION) 
        {
            File file = fileChooser.getSelectedFile();
            String extension = file.toString().substring(file.toString().lastIndexOf(".")+1);

            //the data set we are loading
            ClassificationDataSet tmpDataSet;
            
            if(extension.equalsIgnoreCase("arff"))
            {
                DataSet loaded = ARFFLoader.loadArffFile(file);
                if (loaded.getNumCategoricalVars() == 0)
                {
                    SwingUtilities.invokeLater(() ->
                    {
                        JOptionPane.showMessageDialog(this, 
                                "The loaded dataset dosn't have any categorical features to be the classification target ", 
                                "Loading ARFF File Error", JOptionPane.ERROR_MESSAGE);
                    });
                    return;
                }
                else if(loaded.getNumCategoricalVars() > 1)
                {
                    //TODO instead lets display a dialog and have the user pick which one they want
                    SwingUtilities.invokeLater(() ->
                    {
                        JOptionPane.showMessageDialog(this, 
                                "The loaded dataset dosn't has more than one categorical features, using " + loaded.getCategoryName(loaded.getNumCategoricalVars()-1), 
                                "Loading ARFF, multiple options", JOptionPane.ERROR_MESSAGE);
                    });
                }
                
                tmpDataSet = new ClassificationDataSet(loaded, loaded.getNumCategoricalVars()-1);
            }
            else if(extension.equalsIgnoreCase("txt"))
            {
                /** 
                 * Maps each unique string to its class ID
                 */
                Map<String, Integer> options = new HashMap<>();
                /**
                 * Maps the point index to its end class
                 */
                List<Integer> pointClass = new ArrayList<>();
                List<Vec> pointVecs = new ArrayList<>();
                try
                {
                    BufferedReader br = new BufferedReader(new FileReader(file));
                    String line;
                    while((line = br.readLine()) != null)
                    {
                        String[] split = line.trim().split("[\\s,]+");
                        if(split.length < 3)
                            continue;
                        pointVecs.add(toDenseVec(parseDouble(split[0]),parseDouble(split[1])));
                        //map to a class for the 3rd index
                        split[2] = split[2].trim();
                        if(!options.containsKey(split[2]))
                            options.put(split[2], options.size());
                        pointClass.add(options.get(split[2]));
                    }
                    
                    //Now create the dpList
                    CategoricalData[] catData = new CategoricalData[] { new CategoricalData(options.size()) } ;
                    //convert everything to a data set
                    List<DataPoint> dpList = new ArrayList<>();
                    for(int i = 0; i < pointVecs.size(); i++)
                    {
                        DataPoint dp = new DataPoint(pointVecs.get(i), new int[]{pointClass.get(i)}, catData);
                        dpList.add(dp);
                    }
                    
                    //empty? Means none of the lines looked like data points
                    if(dpList.isEmpty())
                    {
                        SwingUtilities.invokeLater(() ->
                        {
                            JOptionPane.showMessageDialog(this, 
                                    "The text file dosn't appear to be formated as \"#, #, className\", no lines were found matching this pattern", 
                                    "Loading Text File Failure", JOptionPane.ERROR_MESSAGE);
                        });
                    }
                    
                    tmpDataSet = new ClassificationDataSet(new SimpleDataSet(dpList), 0);
                }
                catch(IOException | NumberFormatException ex)
                {
                    SwingUtilities.invokeLater(() ->
                    {
                        JOptionPane.showMessageDialog(this,
                                "The dataset failed to load with the following error:\n" + ex.getMessage(),
                                "Loading Text File Failure", JOptionPane.ERROR_MESSAGE);
                    });
                    return;
                }
            }
            else
            {
                SwingUtilities.invokeLater(() ->
                {
                    JOptionPane.showMessageDialog(this,
                            "An unkown error occured, please report bug",
                            "Unkown Issue", JOptionPane.ERROR_MESSAGE);
                });
                return;
            }

            if(tmpDataSet.getNumNumericalVars() != 2)
            {
                SwingUtilities.invokeLater(() ->
                {
                    JOptionPane.showMessageDialog(this,
                            "The data set that was loaded had more than 2 numerical features.\nThis tool is only meant for 2D problems",
                            "Dataset Issue", JOptionPane.ERROR_MESSAGE);
                });
            }
            
            System.out.println("Loaded, N: " + tmpDataSet.getSampleSize());
            dataSet = tmpDataSet;
            dataSet.applyTransform(new LinearTransform(dataSet));
            
            final JFXPanel fxPanel = new JFXPanel();
            Platform.runLater(() ->
            {
                fxPanel.setScene(new Scene(new BorderPane(Plot.scatterC(dataSet))));
            });
            
            if(centerTabbed != null)
            {
                remove(centerTabbed);
                plotList.clear();
            }
            
            centerTabbed = new JTabbedPane();
            centerTabbed.add("Original Data Set", fxPanel);

            add(centerTabbed, BorderLayout.CENTER);
            getContentPane().validate();
            getContentPane().repaint();
        }
    }//GEN-LAST:event_jMenuItemOpenActionPerformed

    private void jMenuItemPlotResolitionActionPerformed(java.awt.event.ActionEvent evt)//GEN-FIRST:event_jMenuItemPlotResolitionActionPerformed
    {//GEN-HEADEREND:event_jMenuItemPlotResolitionActionPerformed
        String input = JOptionPane.showInputDialog(rootPane, "Please enter render resolution");
        if(input == null)
            return;
        int newRes = Integer.parseInt(input);
        if (newRes <= 0)
        {
            JOptionPane.showMessageDialog(rootPane, "Resolution must be a positive integer");
            return;
        }

        plotResolution = newRes;
        plotList.stream().forEach((plot) -> plot.setResolution(plotResolution));
        
        validate();
        repaint();
    }//GEN-LAST:event_jMenuItemPlotResolitionActionPerformed

    private void jCheckBoxMenuItemHardBoundariesActionPerformed(java.awt.event.ActionEvent evt)//GEN-FIRST:event_jCheckBoxMenuItemHardBoundariesActionPerformed
    {//GEN-HEADEREND:event_jCheckBoxMenuItemHardBoundariesActionPerformed
        plotList.stream().forEach((plot) -> plot.setHardBoundaries(jCheckBoxMenuItemHardBoundaries.isSelected()) );
        validate();
        repaint();
    }//GEN-LAST:event_jCheckBoxMenuItemHardBoundariesActionPerformed

    private void jCheckBoxMenuItemEqualWeightActionPerformed(java.awt.event.ActionEvent evt)//GEN-FIRST:event_jCheckBoxMenuItemEqualWeightActionPerformed
    {//GEN-HEADEREND:event_jCheckBoxMenuItemEqualWeightActionPerformed
        equalWeight = jCheckBoxMenuItemEqualWeight.isSelected();
    }//GEN-LAST:event_jCheckBoxMenuItemEqualWeightActionPerformed

    /**
     * @param args the command line arguments
     */
    public static void main(String args[])
    {
        //For OSX, dosn't impact anyone else - so who cares. 
        System.setProperty("apple.laf.useScreenMenuBar", "true");

        /*
         * Create and display the form
         */
        java.awt.EventQueue.invokeLater(() -> new ClassifierToyWorld().setVisible(true) );
    }
    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JPanel bottomAreaPanel;
    private javax.swing.ButtonGroup buttonGroupMeta;
    private javax.swing.ButtonGroup buttonGroupMultiClass;
    private javax.swing.JCheckBoxMenuItem jCheckBoxMenuItemEqualWeight;
    private javax.swing.JCheckBoxMenuItem jCheckBoxMenuItemHardBoundaries;
    private javax.swing.JCheckBoxMenuItem jCheckBoxMenuItemParallel;
    private javax.swing.JLabel jLabelInfo;
    private javax.swing.JMenuBar jMenuBar1;
    private javax.swing.JMenu jMenuCC;
    private javax.swing.JMenu jMenuClassifiers;
    private javax.swing.JMenu jMenuEdit;
    private javax.swing.JMenu jMenuFile;
    private javax.swing.JMenuItem jMenuItemOpen;
    private javax.swing.JMenuItem jMenuItemPlotResolition;
    private javax.swing.JMenu jMenuMultiClassExtension;
    private javax.swing.JRadioButtonMenuItem jRadioBMetaAdaBoosM1;
    private javax.swing.JRadioButtonMenuItem jRadioBMetaBagging;
    private javax.swing.JRadioButtonMenuItem jRadioBMetaEmphasisBoost;
    private javax.swing.JRadioButtonMenuItem jRadioBMetaModestBoost;
    private javax.swing.JRadioButtonMenuItem jRadioBMetaNone;
    private javax.swing.JRadioButtonMenuItem jRadioBMetaSAMME;
    private javax.swing.JRadioButtonMenuItem jRadioButtonMenuItemMultiClassDDAG;
    private javax.swing.JRadioButtonMenuItem jRadioButtonMenuItemMultiClassNative;
    private javax.swing.JRadioButtonMenuItem jRadioButtonMenuItemMultiClassOneVsAll;
    private javax.swing.JRadioButtonMenuItem jRadioButtonMenuItemMultiClassOneVsOne;
    // End of variables declaration//GEN-END:variables
}
