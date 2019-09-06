//
//  ViewController.swift
//  temp
//
//  Created by Suthar, Bhavin Udavji on 04/06/1941 Saka.
//  Copyright © 1941 Suthar, Bhavin Udavji. All rights reserved.
//

import UIKit
import CoreML
import Vision
class ViewController: UIViewController {
    
    // Model pre-trained on 1,9 and 5 digits
    // retrain it with 7 and save it
    // reload the saved model
    // check if it gives 7,1,9,5 as  predictions
    
    
    var classifierURL = UpdatableMNISTDigitClassifier.urlOfModelInThisBundle
    var modelName = "v2.mlmodelc"
    
    lazy var classifier: MLModel = {
       return UpdatableMNISTDigitClassifier().model
    }()
    
    lazy var modelConfig: MLModelConfiguration = {
        let modelConfig = MLModelConfiguration()
        modelConfig.computeUnits = .all
        return modelConfig
    }()
    
    
    override func viewDidLoad() {
        super.viewDidLoad()
        try! predict(url: classifierURL)
        startTraining()
    }
    
    /// Predict from URL of ml model
    /// - Parameter url:
    func predict(url:URL) throws {
        let classifier = try UpdatableMNISTDigitClassifier.init(contentsOf: url, configuration: modelConfig)
        let image = UIImage.init(named: "seven")!
        let input = UpdatableMNISTDigitClassifierInput.init(image:convert(image: image.cgImage!)  )
        let results =  try! classifier.prediction(input: input)
        print(results.digit)
    }
    /// Predict with ml model
    /// - Parameter model:
    func predict(model:MLModel)  throws {
        let image = UIImage.init(named: "seven")!
        let input = UpdatableMNISTDigitClassifierInput.init(image:convert(image: image.cgImage!)  )
        let testData = MLArrayBatchProvider.init(array: [input])
        let results = try model.predictions(fromBatch: testData)
        print(results.features(at: 0).featureValue(for: "digit"))
    }
    
    func startTraining(){
                
        let trainingValues = getTrainingData()
        
        do {
            let updateTask = try MLUpdateTask(forModelAt: classifierURL, trainingData: trainingValues, configuration: modelConfig,
                                              progressHandlers: MLUpdateProgressHandlers(forEvents: [.trainingBegin,.epochEnd],
                                                                                         progressHandler: { (contextProgress) in
                                                                                            print(contextProgress.event)
                                                                                            
                                              }) { (finalContext) in
                                                if (finalContext.task.error == nil) {
                                                    do {
                                                        //Check if its trained on seven
                                                        try self.predict(model: finalContext.model)
                                                        
                                                        // Save it
                                                        self.saveMode(model: finalContext.model)
                                                        
                                                        // Reload and again test if it gives any correct predictions
                                                        let urlToSave = try self.getModelDirURL(modelName:self.modelName)
                                                        try self.predict(url: urlToSave)
                                                        
                                                    }
                                                    catch {
                                                        print(error)
                                                    }
                                                }
            })
            updateTask.resume()
            
        } catch {
            print("Error while upgrading \(error.localizedDescription)")
        }
    }
    
}

extension ViewController {
    func getModelDirURL(modelName:String) throws -> URL{
        
        let fileManager = FileManager.default
        let appSupportDirectory = try fileManager.url(for: .applicationSupportDirectory,
                                                      in: .userDomainMask, appropriateFor: nil, create: true)
        let permanentUrl = appSupportDirectory.appendingPathComponent(modelName)
        print(permanentUrl)
        return permanentUrl
    }
    
    func saveMode(model:MLWritable){
        
        do {
            let urlToSave = try getModelDirURL(modelName: modelName)
            try model.write(to: urlToSave)
        }
        catch {
            print(error)
        }
    }
    
    
    func convert(image: CGImage)  -> CVPixelBuffer {
        let imageInputDescription = classifier.modelDescription.inputDescriptionsByName["image"]!
        let imageConstraint = imageInputDescription.imageConstraint!
        return try! MLFeatureValue(cgImage: image, constraint: imageConstraint).imageBufferValue!
    }
    
    func getTrainingData() -> MLArrayBatchProvider {
        
        let image = UIImage.init(named:"seven")!
        let images = [image,image,image]
        var dataArr = [CVPixelBuffer]()
        
        for image in images{
            dataArr.append(convert(image: image.cgImage!))
        }
        
        let trainingModels = dataArr.map {  UpdatableMNISTDigitClassifierTrainingInput(image: $0 , digit: "7")}
        
        let modelsBatch = MLArrayBatchProvider(array:trainingModels)
        
        
        //        let mlarr =  try! MLMultiArray.init(shape: [2], dataType: .double)
        //                mlarr[0] = 2
        //                mlarr[1] = 2
        //         let mlarr1 =  try! MLMultiArray.init(shape: [1], dataType: .double)
        //         mlarr1[0] = 4
        //         let trainingData = getTrainingData()
        //
        //         let ip = addupdatableTrainingInput.init(number: mlarr, output_true: mlarr1)
        //        // let modelsBatch = MLArrayBatchProvider(array:[ip])
        
        return modelsBatch
    }
}
