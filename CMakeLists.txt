cmake_minimum_required(VERSION 3.12)

project(CS271-Project)

find_package(Python3 COMPONENTS Interpreter REQUIRED)
find_program(WGET_EXECUTABLE wget)

if(NOT WGET_EXECUTABLE)
    if(UNIX AND NOT APPLE)
        message(STATUS "wget is not installed. Installing wget...")

        execute_process(
            COMMAND id -u
            OUTPUT_VARIABLE USER_ID
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )

        if(USER_ID EQUAL 0)
            execute_process(
                COMMAND apt-get install -y wget
                RESULT_VARIABLE install_result
            )
        else()
            execute_process(
                COMMAND sudo apt-get install -y wget
                RESULT_VARIABLE install_result
            )
        endif()

        if(NOT install_result EQUAL 0)
            message(FATAL_ERROR "Failed to install wget. Please install it manually.")
        else()
            find_program(WGET_EXECUTABLE wget)
            if(NOT WGET_EXECUTABLE)
                message(FATAL_ERROR "wget installation failed. Please install it manually.")
            else()
                message(STATUS "wget installed successfully: ${WGET_EXECUTABLE}")
            endif()
        endif()
    else()
        message(WARNING "wget not found. Please install it manually and add to path.")
    endif()
else()
    message(STATUS "wget found: ${WGET_EXECUTABLE}")
endif()

if(WGET_EXECUTABLE)
    # Verifying dataset
    message(STATUS "verifying dataset in ${CMAKE_SOURCE_DIR}/dataset")

    # Define the directory where all dataset files will be stored.
    set(DATASET_DIR "${CMAKE_SOURCE_DIR}/dataset")
    file(MAKE_DIRECTORY "${DATASET_DIR}")

    # List of individual dataset files to download.
    set(DATASETS
        "att48.tsp"
        "att48_d.txt"
        "att48_xy.txt"
        "att48_s.txt"
        "dantzig42.tsp"
        "dantzig42_d.txt"
        "five_d.txt"
        "five_s.txt"
        "fri26.tsp"
        "fri26_d.txt"
        "fri26_s.txt"
        "gr17.tsp"
        "gr17_d.txt"
        "gr17_s.txt"
        "p01.tsp"
        "p01_d.txt"
        "p01_s.txt"
        "p01_sxy.txt"
        "p01_xy.txt"
    )

    # Loop over each dataset file, download it into the dataset folder if it doesn't already exist.
    foreach(dataset ${DATASETS})
        set(DATASET_URL "https://people.sc.fsu.edu/~jburkardt/datasets/tsp/${dataset}")
        set(DATASET_FILE "${DATASET_DIR}/${dataset}")

        if(NOT EXISTS "${DATASET_FILE}")
            message(STATUS "Downloading ${dataset}...")
            execute_process(
                COMMAND ${WGET_EXECUTABLE} ${DATASET_URL} -q -O "${DATASET_FILE}"
                RESULT_VARIABLE download_result
            )
            if(NOT download_result EQUAL 0)
                message(FATAL_ERROR "Failed to download ${dataset}.")
            else()
                message(STATUS "${dataset} downloaded successfully.")
            endif()
        else()
            message(STATUS "${dataset} already exists.")
        endif()
    endforeach()
else()
    message(WARNING "Skipping dataset check. wget not found.")
endif()

add_custom_target(run
    COMMAND ${Python3_EXECUTABLE} main.py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Running Python script"
)