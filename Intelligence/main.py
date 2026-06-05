import sys
import uvicorn
from src.utils import load_and_clean_data

def print_help():
    print("\n" + "=" * 50)
    print("  Dostupne naredbe (možeš ih i povezati, npr. 'preprocess train'):")
    print("=" * 50)
    print("  eda        -> Pokretanje vizualizacije i analize podataka")
    print("  preprocess -> Čišćenje, Target Encoding i izvoz mapa")
    print("  train      -> Treniranje HGB, MLP i ExtraTrees modela")
    print("  evaluate   -> Generiranje grafova grešaka i krivulja učenja")
    print("  importance -> Izračun važnosti značajki (Feature Importance)")
    print("  pipeline   -> Pokreće SVE korake redom (E2E trening)")
    print("  serve      -> Podizanje FastAPI mikroservisa za ASP.NET")
    print("  help       -> Prikaz ovog izbornika")
    print("  exit / q   -> Izlaz iz programa")
    print("=" * 50)

def execute_commands(commands_list):
    """Izvršava listu naredbi jednu za drugom."""
    if 'serve' in commands_list:
        print("\nZapočinjem FastAPI poslužitelj na portu 8000...")
        uvicorn.run("src.api:app", host="127.0.0.1", port=8000, reload=True)
        return False

    df = None
    if any(cmd in ['eda', 'preprocess', 'pipeline'] for cmd in commands_list):
        try:
            df = load_and_clean_data()
        except Exception as e:
            print(f"❌ Greška pri učitavanju skupa podataka: {e}")
            return True

    for cmd in commands_list:
        if cmd == 'eda':
            from src.eda import run_eda
            run_eda(df)
        elif cmd == 'preprocess':
            from src.preprocessing import run_preprocessing
            run_preprocessing(df)
        elif cmd == 'train':
            from src.train import run_training
            run_training()
        elif cmd == 'evaluate':
            from src.evaluate import run_evaluation
            run_evaluation()
        elif cmd == 'importance':
            from src.importance import run_importance
            run_importance()
        elif cmd == 'pipeline':
            print("\n=== POKREĆEM KOMPLETAN END-TO-END ML PIPELINE ===")
            from src.eda import run_eda
            from src.preprocessing import run_preprocessing
            from src.train import run_training
            from src.evaluate import run_evaluation
            from src.importance import run_importance
            
            run_eda(df)
            run_preprocessing(df)
            run_training()
            run_evaluation()
            run_importance()
            print("\n=== PIPELINE USPJEŠNO IZVRŠEN! ===")
        elif cmd in ['exit', 'q']:
            print("Doviđenja!")
            return False
        elif cmd == 'help':
            print_help()
        else:
            print(f"⚠️ Nepoznata naredba: '{cmd}'. Utipkaj 'help' za popis.")
            
    return True

def main():
    if len(sys.argv) > 1:
        commands = [arg.lower() for arg in sys.argv[1:]]
        execute_commands(commands)
        return

    print("=" * 60)
    print("🚘 Dobrodošao u Car Price Estimator interaktivnu konzolu!")
    print("Unesi jednu ili više naredbi odvojenih razmakom.")
    print("=" * 60)
    print_help()

    while True:
        try:
            user_input = input("\n[ML-Konzola] >> ").strip().lower()
            if not user_input:
                continue
            
            commands = user_input.split()
            
            should_continue = execute_commands(commands)
            if not should_continue:
                break
                
        except (KeyboardInterrupt, EOFError):
            print("\nIzlazim... Doviđenja!")
            break

if __name__ == '__main__':
    main()