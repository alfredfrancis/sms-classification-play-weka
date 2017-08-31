package controllers;

import play.mvc.*;
import play.libs.Json;
import views.html.index;

import java.io.IOException;
import java.util.Map;
import java.util.HashMap;
import java.io.File;

import classifier.WekaClassifier;

public class ClassificationController extends Controller {

    private static final String MODEL = "public/models/sms.dat";
    private WekaClassifier classifier;
    private boolean trained;

    ClassificationController() {
        trained = false;
        classifier = new WekaClassifier();
    }

    public Result index(){
        return ok(index.render());
    }

    public Result train() {
        Map<String, String> result = new HashMap<>();
        try{
            classifier.transform();
            classifier.fit();
            trained = true;
            classifier.saveModel(MODEL);
            result.put("status", "success");
            return ok(Json.toJson(result));
        } catch (RuntimeException e){
            result.put("status", "error");
            result.put("message",e.getMessage());
        }
        catch (IOException e){
            result.put("status", "error");
            result.put("message","Couldn't save Model file.");
        }
        return internalServerError(Json.toJson(result));
    }

    public Result evaluate() {
        Map<String, String> result = new HashMap<>();
        try{
            if(!trained &&  !new File(MODEL).exists())
            {
                throw new RuntimeException("Train model before evaluating.");

            }else if(!trained && new File(MODEL).exists() )
            {
                classifier.loadModel(MODEL);
            }

            return ok(classifier.evaluate());

        } catch (RuntimeException e){
            result.put("status","error");
            result.put("message",e.getMessage());
        }catch (ClassNotFoundException e){
            result.put("status", "error");
            result.put("message","Invalid Model.");
        } catch (IOException e){
            result.put("status", "error");
            result.put("message","Couldn't open saved Model file.");
        }
        return internalServerError(Json.toJson(result));
    }

    public Result predict() {


        Map<String, String> result = new HashMap<>();
        try{
            String message = Controller.request().queryString().get("message")[0];

            if (new File(MODEL).exists()) {
                classifier.loadModel(MODEL);
            } else {
                classifier.transform();
                classifier.fit();
                classifier.saveModel(MODEL);
            }
            result.put("status", "success");
            result.put("message", message);
            result.put("label", classifier.predict(message));
            return ok(Json.toJson(result));
        }
        catch (ClassNotFoundException e){
            result.put("status", "error");
            result.put("message","Invalid Model.");
        }
        catch (IOException e){
            result.put("status", "error");
            result.put("message","Couldn't open saved Model file.");
        }
        catch (NullPointerException e) {
            result.put("status", "error");
            result.put("message", "missing query string message");
        }
        catch (RuntimeException e){
            result.put("status", "error");
            result.put("message", e.getMessage());
        }
        return internalServerError(Json.toJson(result));
    }
}