(require 'cl-lib)

;; Define a mock environment function that returns a fixed transition for each action
(defun mock-env (state action)
  (let ((transitions (list (list (mod (+ state action) 5) 1.0))))
    (list (car (nth action transitions)) (cadr (nth action transitions)) :is-done nil)))

;; Define a test for the argmax function
(ert-deftest test-argmax ()
  (should (equal (argmax [1 2 3 4 5]) 4))
  (should (equal (argmax [5 4 3 2 1]) 0))
  (should (equal (argmax [3 3 3 3 3]) 0)))

;; Define a test for the choose-action function
(ert-deftest test-choose-action ()
  (let ((q-values (make-vector 5 (make-vector 5 0.0))))
    (should (cl-find (choose-action q-values 0 0 0.0 5) (number-sequence 0 4)))
    (should (cl-find (choose-action q-values 0 0 1.0 5) (number-sequence 0 4)))))

;; Define a test for the update-q-values function
(ert-deftest test-update-q-values ()
  (let ((q-values (make-vector 5 (make-vector 5 0.0))))
    (update-q-values q-values 0 0 1.0 1 0.5 0.9)
    (should (equal (aref (aref q-values 0) 0) 0.5))
    (should (equal (aref (aref q-values 0) 1) 0.9))))

;; Define a test for the learn-from-transition function
(ert-deftest test-learn-from-transition ()
  (let ((q-values (make-vector 5 (make-vector 5 0.0))))
    (should (equal (learn-from-transition q-values 0 0 1.0 1 0.5 0.9)
                   (vector
                    (vector 0.0 0.0 0.0 0.0 0.0)
                    (vector 0.5 0.0 0.0 0.0 0.0)
                    (vector 0.0 0.0 0.0 0.0 0.0)
                    (vector 0.0 0.0 0.0 0.0 0.0)
                    (vector 0.0 0.0 0.0 0.0 0.0))))))

;; Define a test for the run-episode function
(ert-deftest test-run-episode ()
  (let ((q-values (make-vector 5 (make-vector 5 0.0))))
    (should (equal (run-episode q-values 'mock-env 5 0.5 0.9 0.1 10)
                   1.0))))

;; Define a test for the train-q-learning-agent function
(ert-deftest test-train-q-learning-agent ()
  (let ((q-values (make-vector 5 (make-vector 5 0.0))))
    (should (vectorp (train-q-learning-agent 'mock-env 5 0.5 0.9 0.1 100 10)))))
