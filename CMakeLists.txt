cmake_minimum_required(VERSION 3.12)

project(CS271-Project)

find_package(Python3 COMPONENTS Interpreter REQUIRED)

if(NOT Python3_EXECUTABLE)
    message(FATAL_ERROR "Python3 interpreter not found!")
endif()

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import venv"
    RESULT_VARIABLE VENV_RESULT
    OUTPUT_QUIET
    ERROR_QUIET
)

if(NOT VENV_RESULT EQUAL 0)
    message(WARNING "python3-venv module is not available on this system!")
else()
    message(STATUS "python3-venv is available.")
endif()


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
        "dantzig42.tsp"
        "fri26.tsp"
        "gr17.tsp"
        "p01.tsp"
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

# credit to https://pikotutorial.com/setting-up-a-python-project-with-cmake/
# define a function for creating Python virtual environment
function(create_venv venv_dir requirements_path)
    # check if the virtual environment already exists
    if(EXISTS ${venv_dir})
        message(STATUS "Virtual environment already exists in ${venv_dir}, skipping creation.")
        return()
    endif()
    # ensure that the given requirements.txt file exists
    if(NOT EXISTS ${requirements_path})
        message(FATAL_ERROR "Requirements file not found: ${requirements_path}")
    endif()
    # create the virtual environment
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -m venv ${venv_dir}
        RESULT_VARIABLE venv_creation_ret_code
    )
    # report error if return code is non-zero
    if(venv_creation_ret_code)
        message(FATAL_ERROR "Failed to create virtual environment at ${venv_dir}!")
    endif()
    # install dependencies from requirements.txt
    execute_process(
        COMMAND ${venv_dir}/bin/pip install -r ${requirements_path}
        RESULT_VARIABLE pip_install_ret_code
    )
    # report error if return code is non-zero
    if(pip_install_ret_code)
        message(FATAL_ERROR "Failed to install dependencies from ${requirements_path}!")
    endif()
    # print success message
    message(STATUS "Virtual environment setup done at ${venv_dir} with dependencies from ${requirements_path}")
endfunction()

if( VENV_RESULT EQUAL 0 )
    # specify app1 virtual environment directory
    set(APP_VENV ${CMAKE_BINARY_DIR}/app_venv)
    # create virtual environment for app1
    create_venv(${APP_VENV} ${CMAKE_SOURCE_DIR}/requirements.txt)
else()
    message(WARNING "Please run 'pip install -r requirements.txt' to install dependencies.")
endif()

add_custom_target(run
    COMMAND ${APP_VENV}/bin/python ${CMAKE_SOURCE_DIR}/main.py
    DEPENDS ${APP_VENV}
)